import unittest
import numpy as np
import timeit
from data import generate_sample_dataset

sample_dataset_size = 1_000_000
sample_test_iterations = 25


class TestCorrelationDirection(unittest.TestCase):

    def test_correlation_direction(self):
        # Generate sample dataset of 10,000 VectorEntity objects
        sample_dataset = generate_sample_dataset(100_000)

        # Extract the skill ratios and direction vectors (gains) for all players
        skill_ratios = np.array([player.skill_ratio_first_scrape for player in sample_dataset])
        gained_skill_ratios = np.array([player.skill_gain_ratio for player in sample_dataset])

        # Step 1: Calculate imbalance (deviation from mean)
        skill_means = np.mean(skill_ratios, axis=1, keepdims=True)
        imbalance_vectors = skill_ratios - skill_means  # Imbalance from the mean

        # Step 2: Normalize the imbalance vectors and the direction (gains) vectors
        # Normalize imbalance vectors
        imbalance_norms = np.linalg.norm(imbalance_vectors, axis=1, keepdims=True)
        normalized_imbalance = np.divide(imbalance_vectors, imbalance_norms, where=imbalance_norms != 0)

        # Normalize the direction vectors (gained skills)
        direction_norms = np.linalg.norm(gained_skill_ratios, axis=1, keepdims=True)
        normalized_direction = np.divide(gained_skill_ratios, direction_norms, where=direction_norms != 0)

        # Step 3: Calculate dot product (cosine similarity) between normalized imbalance and direction vectors
        dot_products = np.sum(normalized_imbalance * normalized_direction, axis=1)

        # Check the range of dot products
        print(f"Dot product range: [{np.min(dot_products)}, {np.max(dot_products)}]")

        # Ensure the output values are between -1 and 1
        self.assertTrue(np.all(dot_products >= -1) and np.all(dot_products <= 1))

        # Example assertion: Check that the test completes successfully (replace this with more specific tests)
        self.assertIsNotNone(dot_products)

    def test_correlation_speed(self):
        # Generate sample dataset of 10,000 VectorEntity objects
        sample_dataset = generate_sample_dataset(sample_dataset_size)

        # Benchmark using timeit to measure performance of correlation calculation
        def calculate_correlation():
            # Extract the skill ratios and direction vectors (gains) for all players
            skill_ratios = np.array([player.skill_ratio_first_scrape for player in sample_dataset])
            gained_skill_ratios = np.array([player.skill_gain_ratio for player in sample_dataset])

            # Step 1: Calculate imbalance (deviation from mean)
            skill_means = np.mean(skill_ratios, axis=1, keepdims=True)
            imbalance_vectors = skill_ratios - skill_means

            # Step 2: Normalize the imbalance vectors and the direction (gains) vectors
            imbalance_norms = np.linalg.norm(imbalance_vectors, axis=1, keepdims=True)
            normalized_imbalance = np.divide(imbalance_vectors, imbalance_norms, where=imbalance_norms != 0)

            direction_norms = np.linalg.norm(gained_skill_ratios, axis=1, keepdims=True)
            normalized_direction = np.divide(gained_skill_ratios, direction_norms, where=direction_norms != 0)

            # Step 3: Calculate dot product (cosine similarity) between normalized imbalance and direction vectors
            np.sum(normalized_imbalance * normalized_direction, axis=1)

        # Time the operation over 100 runs
        time_taken = timeit.timeit(calculate_correlation, number=sample_test_iterations)

        # Print average time per run
        print(f"Avg time per correlation computation: {time_taken / 100:.6f} seconds.")


# Run the test
if __name__ == '__main__':
    print("UnitTest - Numpy")
    unittest.main()
