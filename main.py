import tiktoken
import torch
import json
import matplotlib.pyplot as plt
import platform

from torch.utils.backcompat import keepdim_warning
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path




def RmsNorm(tensor,normWeights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1,keepdim=True)))


def displayQkHeatmap(qkPerToken,promptSplitAsTokens):
    qkPerToken = qkPerToken.to("cpu")
    _, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(qkPerToken.to(float).detach(), cmap='viridis')
    ax.set_xticks(range(len(promptSplitAsTokens)))
    ax.set_yticks(range(len(promptSplitAsTokens)))
    ax.set_xticklabels(promptSplitAsTokens)
    ax.set_yticklabels(promptSplitAsTokens)
    ax.figure.colorbar(im, ax=ax)
    plt.title("Q-K Attention Heatmap")
    plt.show()




def main():

    #Update these path based on your models.
    tokenizerPath = r"llama\checkpoints\Llama3.2-1B\tokenizer.model"
    modelPath = r"llama\checkpoints\Llama3.2-1B"
    currentOS = platform.system()

    print("|---------------------|")
    print("|LLAMA3.2 FROM SCRATCH|")
    print("|_____________________|")
    print("Current OS:" + currentOS)

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
    print("\n"+word)


    #3
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(device)
    model = torch.load(modelPath+"/consolidated.00-002.pth",map_location=device)
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

    if currentOS == "Darwin":
        torch.mps.synchronize()
        print("Q Per Token Shape:\t" + str(qPerToken.shape))
    else:
        print("Q Per Token Shape:\t" + str(qPerToken.shape))

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
    frequenciesCis = torch.polar(
        torch.ones_like(frequenciesForEachToken[:, :32]),  # Only take first 32 frequencies
        frequenciesForEachToken[:, :32]
    )


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
    frequenciesCis = frequenciesCis.to(qPerTokenAsComplexNumbers.device)

    qPerTokenAsComplexNumbersRotated = qPerTokenAsComplexNumbers * frequenciesCis
    print(qPerTokenAsComplexNumbersRotated.shape)


    #17
    qPerTokenSplitIntoPairsRotated = torch.view_as_real(qPerTokenAsComplexNumbersRotated)
    print(qPerTokenSplitIntoPairsRotated.shape)

    #18
    qPerTokenRotated = qPerTokenSplitIntoPairsRotated.view(qPerToken.shape)
    print(qPerTokenRotated.shape)

    #19
    kLayer0 = model["layers.0.attention.wk.weight"]
    kLayer0 = kLayer0.view(nKvHeads,kLayer0.shape[0] // nKvHeads,dim)
    print(kLayer0.shape)

    kLayer0Head0 = kLayer0[0]
    print(kLayer0Head0)

    kPerToken = torch.matmul(tokenEmbeddings,kLayer0Head0.T).to(device)
    print(kPerToken.shape)

    kPerTokenSplitIntoPairs = kPerToken.float().view(kPerToken.shape[0],-1,2)
    print(kPerTokenSplitIntoPairs.shape)

    kPerTokenAsComplexNumbers = torch.view_as_complex(kPerTokenSplitIntoPairs)
    print(kPerTokenAsComplexNumbers.shape)

    kPerTokenSplitIntoPairsRotated = torch.view_as_real(kPerTokenAsComplexNumbers * frequenciesCis)
    print(kPerTokenSplitIntoPairsRotated.shape)

    kPerTokenRotated = kPerTokenSplitIntoPairsRotated.view(kPerToken.shape)
    print(kPerTokenRotated.shape)

    #20 Self Attention
    qkPerToken = torch.matmul(qPerTokenRotated, kPerTokenRotated.T) / (headDim) ** 0.5
    print(qkPerToken.shape)

    #21
    displayQkHeatmap(qPerToken,promptSplitAsTokens)

    #22
    mask = torch.full((len(tokens), len(tokens)), float("-inf"),device = tokens.device)
    mask = torch.triu(mask,diagonal=1)
    mask = mask.to(device)
    print(mask)

    #23
    qkPerTokenAfterMasking = qkPerToken + mask
    displayQkHeatmap(qkPerTokenAfterMasking,promptSplitAsTokens)

    #24
    qkPerTokenAfterMaskingAfterSoftmax = torch.nn.functional.softmax(qkPerTokenAfterMasking,dim=1).to(torch.bfloat16)
    displayQkHeatmap(qkPerTokenAfterMaskingAfterSoftmax,promptSplitAsTokens)


    #25
    vLayer0 = model["layers.0.attention.wv.weight"]
    vLayer0 = vLayer0.view(nKvHeads,vLayer0.shape[0] // nKvHeads,dim)
    print(vLayer0.shape)
    vLayer0Head0 = vLayer0[0]
    print(vLayer0Head0.shape)

    #26
    vPerToken = torch.matmul(tokenEmbeddings,vLayer0Head0.T).to(device)
    print(vPerToken.shape)

    #27 Attention
    qkvAttention = torch.matmul(qkPerTokenAfterMaskingAfterSoftmax,vPerToken)
    print(qkvAttention.shape)

    #28 Multi head attention
    qkvAttentionStore = []
    for head in range(numberOfAttentionHeads):
        qLayer0Head = qLayer0[head]
        kLayer0Head = kLayer0[head//4]
        vLayer0Head = vLayer0[head//4]
        qPerToken = torch.matmul(tokenEmbeddings,qLayer0Head.T)
        kPerToken = torch.matmul(tokenEmbeddings,kLayer0Head.T)
        vPerToken = torch.matmul(tokenEmbeddings,vLayer0Head.T)





if __name__ == "__main__":
    main()

