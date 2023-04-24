from sentence_transformers import SentenceTransformer, util
from statistics import mean

responses = [
    [
        "Design patterns are reusable solutions to common software design problems.", 
        "They provide proven templates and guidelines for creating flexible and maintainable software systems.", 
        "The use of design patterns can improve code quality, reduce development time and make software easier to understand and modify."
    ], 
    [
        "Design patterns are reusable solutions to common software design problems.", 
        "They provide proven and tested approaches to address recurring issues.", 
        "They help developers create code that is more flexible, maintainable, and efficient."
    ]
]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = [], embedding = []
for i in range(2):
    for j in range(3):
        embedding.append(model.encode(responses[i][j], convert_to_tensor=True))
    embeddings.append(embedding)
    embedding = []

# compare each sentence in the first response to each sentence in the second response
# ex. responses[0][0] <=> responses[1][0], responses[0][1] <=> responses[1][0], responses[0][2] <=> responses[1][0], ...
outputs = [], output = []
for i in range(3):
    for j in range(3):
        output.append(util.pytorch_cos_sim(embeddings[0][j], embeddings[1][i]).item())
    outputs.append(output)
    output = []

# take an average of the max score for each sentence comparison
maxes = []
for _ in range(3):
    maxes.append(max(outputs[_]))

similarity_score = mean(maxes)
print(similarity_score) # output: 0.7011820077896118

'''
The metric to asses overall similarity used here is to calculate an average of maxes
Compare each sentence in the first response to each sentence in the second response
Take the max of these comparisons. So the first value in the list of maxes will be the highst similarity value of the first sentence 
    of the second response with a sentence in the first response. Essentially, compare each sentence to each sentence and figure out
    the sentence that is the most similar to each sentence in the second response
Take the average of these maxes. This will give an overall similarity of the whole sentences.
This metric is a "Mean of Maxes" approach
'''