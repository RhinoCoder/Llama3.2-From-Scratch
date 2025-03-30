from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt
from torch.utils.backcompat import keepdim_warning

tokenizerPath = "/Users/teatone/.llama/checkpoints/Llama3.2-1B/tokenizer.model"
modelPath = "/Users/teatone/.llama/checkpoints/Llama3.2-1B"

#1
tokenizer_path = tokenizerPath
special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

#2
word = tokenizer.decode(tokenizer.encode("hello world!"))
print(word)


#3
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = torch.load(modelPath+"/consolidated.00.pth",map_location=device)
print(json.dumps(list(model.keys())[:20], indent=4))

#4
with open(modelPath+"/params.json","r") as f:
    config = json.load(f)
    print(config)

#5
dim = config["dim"]
numberOfTransformerLayers = config["n_layers"]
numberOfAttentionHeads = config["n_heads"]
nKvHeads = config["n_kv_heads"]
vocabSize = config["vocab_size"]
multipleOf = config["multiple_of"]
ffnDimMultiplier = config["ffn_dim_multiplier"]
normEps = config["norm_eps"]
ropeTheta = torch.tensor(config["rope_theta"])

#6
#Converting text to tokens.
prompt = "The answer to the ultimate question of life, the universe, and everything is "
tokens = [128000] + tokenizer.encode(prompt)
print(tokens)
tokens = torch.tensor(tokens)
promptSplitAsTokens = [tokenizer.decode([token.item()]) for token in tokens]
print(promptSplitAsTokens)


#7
embeddingLayer = torch.nn.Embedding(vocabSize,dim)
embeddingLayer.weight.data.copy_(model["tok_embeddings.weight"])
tokenEmbeddingsUnnormalized = embeddingLayer(tokens).to(torch.bfloat16)
print(tokenEmbeddingsUnnormalized.shape)

#8 RMS
def RmsNorm(tensor,normWeights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1,keepdim=True)))

#9 First layer of transformer
tokenEmbeddings = RmsNorm(tokenEmbeddingsUnnormalized,model["layers.0.attention_norm.weight"])
print(tokenEmbeddings.shape)


#10
print(
    model["layers.0.attention.wq.weight"].shape,
    model["layers.0.attention.wk.weight"].shape,
    model["layers.0.attention.wv.weight"].shape,
    model["layers.0.attention.wo.weight"].shape
)

#11
qLayer0 = model["layers.0.attention.wq.weight"]
headDim = qLayer0.shape[0] // numberOfAttentionHeads
qLayer0 = qLayer0.view(numberOfAttentionHeads,headDim,dim)
print(qLayer0.shape)

#12
qLayer0Head0 = qLayer0[0]
print(qLayer0Head0.shape)

#13
tokenEmbeddings = tokenEmbeddings.to(device)
qLayer0Head0 = qLayer0Head0.to(device)
qPerToken = torch.matmul(tokenEmbeddings,qLayer0Head0.T).to(device)
torch.mps.synchronize()
print("Q Per Token Shape:\t" +str(qPerToken.shape))

#14 Position Encoding
qPerTokenSplitIntoPairs = qPerToken.float().view(qPerToken.shape[0],-1,2)
print(qPerTokenSplitIntoPairs.shape)

#15
zeroToOneSplitInto64Parts = torch.tensor(range(64)) / 64
print(zeroToOneSplitInto64Parts)
frequencies = 1.0 / (ropeTheta ** zeroToOneSplitInto64Parts)
print("Frequencies: " + str(frequencies))

frequenciesForEachToken = torch.outer(torch.arange(17),frequencies)
frequenciesCis = torch.polar(torch.ones_like(frequenciesForEachToken),frequenciesForEachToken)
print(frequenciesCis.shape)

value = frequenciesCis[3]
plt.figure()
for i,element in enumerate(value[:17]):
    plt.plot([0,element.real],[0,element.imag],color='blue',linewidth=1,label=f"Index: {i}")
    plt.annotate(f"{i}",xy = (element.real,element.imag),color = 'red' )

plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Plot of the one row of frequenciesCis')
plt.show()

#16
qPerTokenAsComplexNumbers = torch.view_as_complex(qPerTokenSplitIntoPairs)
print(qPerTokenAsComplexNumbers.shape)
