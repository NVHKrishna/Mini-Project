from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["model_db"]
collection = db["predictions"]

print("Documents in 'predictions':", collection.count_documents({}))
for doc in collection.find():
    print(doc)
