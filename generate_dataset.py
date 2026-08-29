import random, csv

random.seed(42)

FAKE_TEMPLATES = [
    "This product is AMAZING!!! Best purchase ever!!! 10/10!!!",
    "WOW just WOW!! Everyone should buy this right now!!!",
    "Best product in the world!! Buy it buy it buy it!!!",
    "Amazing amazing amazing!!! Worth every penny!!!",
    "I love this so much!!! Ordered 5 more for my family!!!",
    "This is the BEST thing I have ever bought in my LIFE!!!",
    "PERFECT product!!! No complaints at all!!! Buy NOW!!!",
    "Great great great!!! Fast delivery great quality!!!"
]

REAL_TEMPLATES = [
    "Works as described, though the packaging was slightly damaged. Decent for the price.",
    "Bought this in March. Battery lasts about 6 hours, a bit less than advertised but acceptable.",
    "It's okay for casual use. The material feels a little cheap but does the job.",
    "Delivery took 9 days. Product quality is average, nothing special but not bad either.",
    "Used it for 3 weeks now. The button sticks sometimes, otherwise fine.",
    "Good value for money. Not the best quality but serves my purpose well.",
    "Sound quality is decent. Slight distortion at high volume, overall satisfied.",
    "Bought it after reading reviews. It's fine, though I expected better finishing."
]

rows = []
for _ in range(500):
    text = random.choice(FAKE_TEMPLATES)
    rating = random.choice([5, 5, 5, 1])
    burst = round(random.uniform(0.7, 1.0), 2)
    verified = random.choice([0, 0, 0, 1])
    rows.append([text, rating, burst, verified, 1])

for _ in range(500):
    text = random.choice(REAL_TEMPLATES)
    rating = random.randint(1, 5)
    burst = round(random.uniform(0.0, 0.3), 2)
    verified = random.choice([1, 1, 1, 0])
    rows.append([text, rating, burst, verified, 0])

random.shuffle(rows)
with open("dataset.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["review", "rating", "burst_score", "verified", "label"])
    w.writerows(rows)

print("dataset.csv created with 1000 rows")
