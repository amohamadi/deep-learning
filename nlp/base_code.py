



content = "the king sat on the royal throne"

def create_dataset(content, vocab_size):
    words = content.split()
    
    result = []
    for index in range(len(words)):
        if index + vocab_size < len(words):
            result.append((words[index + vocab_size], tuple(words[index + i] for i in range(vocab_size))))
    
    return result 
        
        
data = create_dataset(content, vocab_size=3)

for item in data:
    print(item)