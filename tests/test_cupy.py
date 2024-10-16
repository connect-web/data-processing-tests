import unittest
import cupy as cp
import timeit
from data import generate_sample_dataset

sample_dataset_size = 1_000_000
sample_test_iterations = 25

EPSILON = 5e-9  # Small value to replace zero norms
# This value was determined as 1 / 200_000_000 since 1 is the smallest experience and 200_000_000 is the largest.

class TestCorrelationDirectionCuPy(unittest.TestCase):

    def test_correlation_direction(self):
        # Generate sample dataset of 10,000 VectorEntity objects
        sample_dataset = generate_sample_dataset(10_000)

        # Extract the skill ratios and direction vectors (gains) for all players
        skill_ratios = cp.array([player.skill_ratio_first_scrape for player in sample_dataset])
        gained_skill_ratios = cp.array([player.skill_gain_ratio for player in sample_dataset])

        # Step 1: Calculate imbalance (deviation from mean)
        skill_means = cp.mean(skill_ratios, axis=1, keepdims=True)
        imbalance_vectors = skill_ratios - skill_means  # Imbalance from the mean

        # Step 2: Normalize the imbalance vectors and the direction (gains) vectors
        # Normalize imbalance vectors, avoiding division by zero by replacing 0 with a small epsilon
        imbalance_norms = cp.linalg.norm(imbalance_vectors, axis=1, keepdims=True)
        imbalance_norms = cp.where(imbalance_norms == 0, EPSILON, imbalance_norms)  # Replace zeros with epsilon
        normalized_imbalance = imbalance_vectors / imbalance_norms

        # Normalize the direction vectors (gained skills), avoiding division by zero
        direction_norms = cp.linalg.norm(gained_skill_ratios, axis=1, keepdims=True)
        direction_norms = cp.where(direction_norms == 0, EPSILON, direction_norms)  # Replace zeros with epsilon
        normalized_direction = gained_skill_ratios / direction_norms

        # Step 3: Calculate dot product (cosine similarity) between normalized imbalance and direction vectors
        dot_products = cp.sum(normalized_imbalance * normalized_direction, axis=1)

        # Synchronize and move the results to CPU for inspection (if needed)
        dot_products_cpu = cp.asnumpy(dot_products)

        # Check the range of dot products
        print(f"Dot product range: [{cp.min(dot_products):.3f}, {cp.max(dot_products):.3f}]")

        # Ensure the output values are between -1 and 1
        self.assertTrue(cp.all(dot_products >= -1) and cp.all(dot_products <= 1))

        # Example assertion: Check that the test completes successfully
        self.assertIsNotNone(dot_products)

    def test_correlation_speed(self):
        # Generate sample dataset of 10,000 VectorEntity objects
        sample_dataset = generate_sample_dataset(sample_dataset_size)

        # Benchmark using timeit to measure performance of correlation calculation
        def calculate_correlation():
            # Extract the skill ratios and direction vectors (gains) for all players
            skill_ratios = cp.array([player.skill_ratio_first_scrape for player in sample_dataset])
            gained_skill_ratios = cp.array([player.skill_gain_ratio for player in sample_dataset])

            # Step 1: Calculate imbalance (deviation from mean)
            skill_means = cp.mean(skill_ratios, axis=1, keepdims=True)
            imbalance_vectors = skill_ratios - skill_means

            # Step 2: Normalize the imbalance vectors and the direction (gains) vectors
            imbalance_norms = cp.linalg.norm(imbalance_vectors, axis=1, keepdims=True)
            imbalance_norms = cp.where(imbalance_norms == 0, EPSILON, imbalance_norms)  # Replace zeros with epsilon
            normalized_imbalance = imbalance_vectors / imbalance_norms

            direction_norms = cp.linalg.norm(gained_skill_ratios, axis=1, keepdims=True)
            direction_norms = cp.where(direction_norms == 0, EPSILON, direction_norms)  # Replace zeros with epsilon
            normalized_direction = gained_skill_ratios / direction_norms

            # Step 3: Calculate dot product (cosine similarity) between normalized imbalance and direction vectors
            cp.sum(normalized_imbalance * normalized_direction, axis=1)

        # Time the operation over 100 runs
        time_taken = timeit.timeit(calculate_correlation, number=sample_test_iterations)

        # Print average time per run
        print(f"Avg time per correlation computation: {time_taken / 100:.6f} seconds.")


# Run the test
if __name__ == '__main__':
    print("UnitTest - CuPy")
    unittest.main()
