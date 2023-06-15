from transformers import BertTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_subjects(subjects):
    # Find the maximum length of the tokenized subjects
    max_length = max(len(tokenizer.encode(subject)) for subject in subjects)
    
    tokenized_subjects = []
    for subject in subjects:
        # Tokenize the subject using BERT tokenizer
        encoded = tokenizer.encode(subject, add_special_tokens=True)
        
        # Pad the encoded sequence with zeros to match the maximum length
        padded = encoded + [0] * (max_length - len(encoded))
        
        # Convert the padded sequence to a NumPy array
        encoded_np = np.asarray(padded)
        
        tokenized_subjects.append(encoded_np)
    
    return tokenized_subjects

def kmeans_clustering(tokenized_subjects, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(tokenized_subjects)
    return kmeans.labels_

def calculate_cosine_similarity(subject1, subject2):
    encoded_subjects = tokenizer([subject1, subject2], truncation=True, padding=True, return_tensors='np')['input_ids']
    cosine_sim = cosine_similarity(encoded_subjects)[0, 1]
    return cosine_sim

def group_subjects(subjects, labels, similarity_threshold):
    grouped_subjects = {}
    for subject, label in zip(subjects, labels):
        if label not in grouped_subjects:
            grouped_subjects[label] = [subject]
        else:
            similar_subjects = grouped_subjects[label]
            is_similar = False
            for sim_sub in similar_subjects:
                similarity = calculate_cosine_similarity(subject, sim_sub)
                if similarity >= similarity_threshold:
                    is_similar = True
                    break
            if is_similar:
                grouped_subjects[label].append(subject)
            else:
                grouped_subjects[label] = [subject]
    return list(grouped_subjects.values())

# Example usage
subjects = [
    "Hello",
    "Hello world",
    "Hi there",
    "Hey",
    "Hey there",
    "Good morning",
    "Good evening",
    "Greetings",
    "Hello, how are you?",
    "Hey, what's up?"
]

# Tokenize the subjects using BERT tokenizer and pad them
tokenized_subjects = tokenize_subjects(subjects)

# Apply K-means clustering to the tokenized subjects
labels = kmeans_clustering(tokenized_subjects, n_clusters=3)

# Group the subjects based on cosine similarity and cluster labels
grouped_subjects = group_subjects(subjects, labels, similarity_threshold=0.8)

# Print the grouped subjects
for group in grouped_subjects:
    print(group)
