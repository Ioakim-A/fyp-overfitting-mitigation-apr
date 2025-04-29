from bert_serving.client import BertClient
m = BertClient(ip='127.0.0.1',port=5555, check_length=False, check_version=False)
print("Connected!")