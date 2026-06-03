import random
import math
from collections import defaultdict
import pandas as pand
import matplotlib.pyplot as plot


# 1. Define global weights to use in the project
 
DATASET_FILE = "ga_exam_timetable_dataset.xlsx"
RANDOM_SEED = 7

POPULATION_SIZE = 300
GENERATIONS = 1500
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.10
ELITE_COUNT = 3

# Penalty weights: higher means more important
SAME_SLOT_CONFLICT_WEIGHT = 1000       # hard: same student has two exams at same time
MORE_THAN_TWO_PER_DAY_WEIGHT = 800     # hard: student has more than 2 exams/day
FOUR_IN_TWO_DAYS_WEIGHT = 500          # hard/strong: 4 exams in two consecutive days
TWO_IN_SAME_DAY_WEIGHT = 100           # soft: preferably avoid two exams in same day
EXTRA_DAY_WEIGHT = 300                 # soft: minimize number of exam days
SLOT_OVERLOAD_WEIGHT = 300             # hard-ish: more than 3 courses in same day/slot is not used here, slots are fixed
PREFERRED_MAX_USED_DAYS = 5


 # 2. Loading and preprocessing exams information
class ExamData:
    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.courses = []
        self.course_index = {}
        self.slot_ids = []
        self.slot_day = []
        self.slot_number = []
        self.slot_time = []
        self.student_courses = {}
        self.conflict_matrix = []
        self.load_dataset()
        self.build_conflict_matrix()

    def load_dataset(self):
        courses_df = pand.read_excel(self.dataset_file, sheet_name="Course_Catalog")
        slots_df = pand.read_excel(self.dataset_file, sheet_name="Exam_Slots")
        students_df = pand.read_excel(self.dataset_file, sheet_name="Student_Courses")

        self.courses = list(courses_df["Course_Code"].dropna().astype(str))
        self.course_index = {course: i for i, course in enumerate(self.courses)}

        self.slot_ids = list(slots_df["Slot_ID"].dropna().astype(str))
        self.slot_day = list(slots_df["Exam_Day"].dropna().astype(int))
        self.slot_number = list(slots_df["Slot_Number"].dropna().astype(int))
        self.slot_time = list(slots_df["Time"].dropna().astype(str))

        for _, row in students_df.iterrows():
            student_id = str(row["Student_ID"])
            taken = []
            for col in ["Course_1", "Course_2", "Course_3", "Course_4", "Course_5", "Course_6", "Course_7"]:
                if col in students_df.columns and pand.notna(row[col]):
                    course = str(row[col])
                    if course in self.course_index:
                        taken.append(course)
            self.student_courses[student_id] = taken

    def build_conflict_matrix(self):
        n = len(self.courses)
        self.conflict_matrix = [[0 for _ in range(n)] for _ in range(n)]

        for courses in self.student_courses.values():
            for i in range(len(courses)):
                for j in range(i + 1, len(courses)):
                    a = self.course_index[courses[i]]
                    b = self.course_index[courses[j]]
                    self.conflict_matrix[a][b] += 1
                    self.conflict_matrix[b][a] += 1


# 3. Chromosome and fitness

class GeneticExamScheduler:
    def __init__(self, data):
        self.data = data
        self.num_courses = len(data.courses)
        self.num_slots = len(data.slot_ids)

    def create_random_chromosome(self):
        return [random.randint(0, self.num_slots - 1) for _ in range(self.num_courses)]

    def decode(self, chromosome):
        rows = []
        for course_i, slot_i in enumerate(chromosome):
            rows.append({
                "Course_Code": self.data.courses[course_i],
                "Assigned_Slot_ID": self.data.slot_ids[slot_i],
                "Exam_Day": self.data.slot_day[slot_i],
                "Slot_Number": self.data.slot_number[slot_i],
                "Time": self.data.slot_time[slot_i]
            })
        rows.sort(key=lambda r: (r["Exam_Day"], r["Slot_Number"], r["Course_Code"]))
        return rows

    def objective_penalty(self, chromosome):
        penalty = 0
        details = defaultdict(int)

        # Constraint 1: same student cannot have two exams at the same time.
        for i in range(self.num_courses):
            for j in range(i + 1, self.num_courses):
                common_students = self.data.conflict_matrix[i][j]
                if common_students > 0 and chromosome[i] == chromosome[j]:
                    penalty += SAME_SLOT_CONFLICT_WEIGHT * common_students
                    details["same_slot_conflicts"] += common_students

        # Students daily constraints.
        for student_id, courses in self.data.student_courses.items():
            exams_per_day = defaultdict(int)

            for course in courses:
                course_i = self.data.course_index[course]
                slot_i = chromosome[course_i]
                day = self.data.slot_day[slot_i]
                exams_per_day[day] += 1

            for day, count in exams_per_day.items():
                # Soft: preferably no two exams in the same day.
                if count == 2:
                    penalty += TWO_IN_SAME_DAY_WEIGHT
                    details["two_exams_same_day"] += 1

                # Hard: more than two exams in the same day.
                if count > 2:
                    penalty += MORE_THAN_TWO_PER_DAY_WEIGHT * (count - 2)
                    details["more_than_two_per_day"] += count - 2

            # Constraint: not four exams in two consecutive days.
            used_days = sorted(exams_per_day.keys())
            for day in used_days:
                two_day_count = exams_per_day.get(day, 0) + exams_per_day.get(day + 1, 0)
                if two_day_count >= 4:
                    penalty += FOUR_IN_TWO_DAYS_WEIGHT * (two_day_count - 3)
                    details["four_in_two_days"] += 1

        # Minimize number of used days.
        used_days_count = len(set(self.data.slot_day[slot_i] for slot_i in chromosome))
        if used_days_count > PREFERRED_MAX_USED_DAYS:
            penalty += EXTRA_DAY_WEIGHT * (used_days_count - PREFERRED_MAX_USED_DAYS)
            details["extra_days"] += used_days_count - PREFERRED_MAX_USED_DAYS

        return penalty, details

    def fitness(self, chromosome):
        penalty, _ = self.objective_penalty(chromosome)
        return 1.0 / (1.0 + penalty)



# 4. GA operators

def initialize_population(scheduler, size):
    return [scheduler.create_random_chromosome() for _ in range(size)]


def stochastic_remainder_selection(population, fitness_values):
    pop_size = len(population)
    total_fitness = sum(fitness_values)

    if total_fitness == 0:
        return random.sample(population, pop_size)

    selected = []
    fractional_candidates = []

    for chromosome, fit in zip(population, fitness_values):
        probability = fit / total_fitness
        expected_count = probability * pop_size
        guaranteed_count = int(expected_count)
        fraction = expected_count - guaranteed_count

        for _ in range(guaranteed_count):
            selected.append(chromosome[:])

        fractional_candidates.append((chromosome, fraction))

    random.shuffle(fractional_candidates)
    for chromosome, fraction in fractional_candidates:
        if len(selected) >= pop_size:
            break
        if random.random() < fraction:
            selected.append(chromosome[:])

    while len(selected) < pop_size:
        selected.append(random.choice(population)[:])

    return selected[:pop_size]


def two_point_crossover(parent1, parent2):
    if len(parent1) < 3:
        return parent1[:], parent2[:]

    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)

    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2


def mutate_random_reset(chromosome, num_slots, mutation_rate):
    child = chromosome[:]
    for i in range(len(child)):
        if random.random() < mutation_rate:
            old_slot = child[i]
            new_slot = random.randint(0, num_slots - 1)
            while new_slot == old_slot and num_slots > 1:
                new_slot = random.randint(0, num_slots - 1)
            child[i] = new_slot
    return child


# 5. GA main loop

def run_ga(dataset_file=DATASET_FILE,
           population_size=POPULATION_SIZE,
           generations=GENERATIONS,
           crossover_rate=CROSSOVER_RATE,
           mutation_rate=MUTATION_RATE,
           elite_count=ELITE_COUNT,
           seed=RANDOM_SEED):

    random.seed(seed)
    data = ExamData(dataset_file)
    scheduler = GeneticExamScheduler(data)
    population = initialize_population(scheduler, population_size)

    best_chromosome = None
    best_penalty = math.inf
    best_details = None
    history = []

    for gen in range(generations + 1):
        evaluated = []
        for chromosome in population:
            penalty, details = scheduler.objective_penalty(chromosome)
            fit = 1.0 / (1.0 + penalty)
            evaluated.append((penalty, fit, chromosome, details))

        evaluated.sort(key=lambda x: x[0])
        current_best_penalty, current_best_fit, current_best_chromosome, current_best_details = evaluated[0]

        if current_best_penalty < best_penalty:
            best_penalty = current_best_penalty
            best_chromosome = current_best_chromosome[:]
            best_details = dict(current_best_details)
        
        if best_penalty == 0:
            print("Perfect solution found.")
            break

        history.append({
            "generation": gen,
            "best_penalty": best_penalty,
            "current_best_penalty": current_best_penalty,
            "average_penalty": sum(item[0] for item in evaluated) / len(evaluated),
            "best_fitness": 1.0 / (1.0 + best_penalty)
        })

        if gen % 25 == 0 or gen == generations:
            print(f"Generation {gen:4d} | best penalty = {best_penalty:8.2f} | current best = {current_best_penalty:8.2f}")

        if gen == generations:
            break

        # keep the best chromosomes unchanged.
        new_population = [item[2][:] for item in evaluated[:elite_count]]

        fitness_values = [item[1] for item in evaluated]
        sorted_population = [item[2] for item in evaluated]
        mating_pool = stochastic_remainder_selection(sorted_population, fitness_values)
        random.shuffle(mating_pool)

        index = 0
        while len(new_population) < population_size:
            parent1 = mating_pool[index % len(mating_pool)]
            parent2 = mating_pool[(index + 1) % len(mating_pool)]
            index += 2

            if random.random() < crossover_rate:
                child1, child2 = two_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = mutate_random_reset(child1, scheduler.num_slots, mutation_rate)
            child2 = mutate_random_reset(child2, scheduler.num_slots, mutation_rate)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    final_schedule = scheduler.decode(best_chromosome)
    return scheduler, best_chromosome, best_penalty, best_details, history, final_schedule


# 6. Saving output files

def save_results(scheduler, best_chromosome, best_penalty, best_details, history, final_schedule):
    schedule_df = pand.DataFrame(final_schedule)
    history_df = pand.DataFrame(history)

    schedule_df.to_csv("best_exam_schedule.csv", index=False)
    history_df.to_csv("convergence_history.csv", index=False)

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write("Genetic Algorithm Exam Scheduler Results\n")
        f.write("========================================\n")
        f.write(f"Best penalty: {best_penalty}\n")
        f.write(f"Best chromosome: {best_chromosome}\n")
        f.write(f"Violation details: {best_details}\n\n")
        f.write("Decoded Schedule:\n")
        for row in final_schedule:
            f.write(f"{row['Course_Code']} -> {row['Assigned_Slot_ID']} "
                    f"(Day {row['Exam_Day']}, Slot {row['Slot_Number']}, {row['Time']})\n")

    if plot is not None:
        plot.figure(figsize=(8, 5))
        plot.plot(history_df["generation"], history_df["best_penalty"], label="Best penalty")
        plot.plot(history_df["generation"], history_df["average_penalty"], label="Average penalty")
        plot.xlabel("Generation")
        plot.ylabel("Penalty")
        plot.title("GA Convergence Rate")
        plot.legend()
        plot.grid(True)
        plot.tight_layout()
        plot.savefig("convergence_plot.png", dpi=150)


def test_sample_bad_schedule(dataset_file=DATASET_FILE):
    data = ExamData(dataset_file)
    scheduler = GeneticExamScheduler(data)
    bad_df = pand.read_excel(dataset_file, sheet_name="Sample_Bad_Schedule")

    slot_to_index = {slot_id: i for i, slot_id in enumerate(data.slot_ids)}
    chromosome = [0] * len(data.courses)

    for _, row in bad_df.iterrows():
        course = str(row["Course_Code"])
        slot_id = str(row["Assigned_Slot_ID"])
        if course in data.course_index and slot_id in slot_to_index:
            chromosome[data.course_index[course]] = slot_to_index[slot_id]

    penalty, details = scheduler.objective_penalty(chromosome)
    print("Sample bad schedule penalty:", penalty)
    print("Sample bad schedule details:", dict(details))


if __name__ == "__main__":
    test_sample_bad_schedule(DATASET_FILE)

    scheduler, best_chromosome, best_penalty, best_details, history, final_schedule = run_ga(
        dataset_file=DATASET_FILE,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        elite_count=ELITE_COUNT,
        seed=RANDOM_SEED
    )

    save_results(scheduler, best_chromosome, best_penalty, best_details, history, final_schedule)

    print("\nFinal best penalty:", best_penalty)
    print("Best chromosome:", best_chromosome)
    print("Violation details:", best_details)
    print("\nFinal decoded schedule:")
    for row in final_schedule:
        print(f"{row['Course_Code']:10s} -> {row['Assigned_Slot_ID']} | Day {row['Exam_Day']} | Slot {row['Slot_Number']} | {row['Time']}")
    print("\nFiles saved: best_exam_schedule.csv, convergence_history.csv, summary.txt, convergence_plot.png")
