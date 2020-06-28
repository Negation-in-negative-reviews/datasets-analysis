
import numpy as np

if __name__ == "__main__":
    seed_val = 23
    np.random.seed(seed_val)
    n_samples = int(1e2)
    filepath = "/data/madhu/amazon-reviews-2018/processed_data/pet_supplies/pos_reviews_train"
    fin = open(filepath, "r")

    reviews = fin.readlines()

    indices = np.random.choice(np.arange(len(reviews)), size=n_samples)

    sampled_reviews = [reviews[idx] for idx in indices]

    for rev in sampled_reviews:
        print(rev.strip("\n"))