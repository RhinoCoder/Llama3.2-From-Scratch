import time
import tiktoken
import torch
import json
import os
import sys
import matplotlib.pyplot as plt
import platform
import torch.nn.functional as F

from torch.nn.functional import rms_norm
from torch.utils.backcompat import keepdim_warning
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path



def clearPreviousOutput(text_length=None):

    if text_length is None:
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()
        return

    try:
        terminal_width = os.get_terminal_size().columns
        lines_needed = max(1, (text_length // terminal_width) + 1)
        sys.stdout.write('\r')

        for _ in range(lines_needed):
            sys.stdout.write('\033[K')
            sys.stdout.write('\033[1A')

        sys.stdout.write('\033[{}B'.format(lines_needed))
        sys.stdout.write('\r')
        sys.stdout.flush()

    except (AttributeError, OSError):
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()


def updateLine(text, previous_length=0):
    if previous_length > 0:
        clearPreviousOutput(previous_length)
    sys.stdout.write('\r' + text)
    sys.stdout.flush()
    return len(text)


def generateSequenceStream(model, initial_tokens, config, tokenizer, max_new_tokens, device):
    tokens = initial_tokens.clone().to(device)
    generated_text = tokenizer.decode(tokens.tolist())

    display_text = "Generated Sequence: " + generated_text
    prev_length = updateLine(display_text)

    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    rope_theta = config["rope_theta"]
    head_dim = dim // n_heads

    for i in range(max_new_tokens):
        embeddingLayer = torch.nn.Embedding(config["vocab_size"], dim).to(device)
        embeddingLayer.weight.data.copy_(model["tok_embeddings.weight"].to(device))
        tokenEmbeddingsUnnormalized = embeddingLayer(tokens).to(torch.bfloat16)
        seqLen = tokens.shape[0]
        freqsCis = precomputeFreqsCis(head_dim, seqLen, device, rope_theta)
        finalEmbedding = tokenEmbeddingsUnnormalized

        for layer in range(n_layers):
            qkvAttentionStore = []
            weightAttentionNorm = model[f"layers.{layer}.attention_norm.weight"].to(device)
            layerEmbeddingNorm = RmsNorm(finalEmbedding, weightAttentionNorm, 1e-5)
            qLayer = model[f"layers.{layer}.attention.wq.weight"].to(device).view(n_heads, -1, dim)
            kLayer = model[f"layers.{layer}.attention.wk.weight"].to(device).view(n_kv_heads, -1, dim)
            vLayer = model[f"layers.{layer}.attention.wv.weight"].to(device).view(n_kv_heads, -1, dim)

            for head in range(n_heads):
                qLayerHead = qLayer[head]
                kLayerHead = kLayer[head // 4]
                vLayerHead = vLayer[head // 4]
                qPerToken = torch.matmul(layerEmbeddingNorm, qLayerHead.T)
                kPerToken = torch.matmul(layerEmbeddingNorm, kLayerHead.T)
                vPerToken = torch.matmul(layerEmbeddingNorm, vLayerHead.T)
                qPerTokenSplitIntoPairs = qPerToken.float().view(qPerToken.shape[0], -1, 2)
                qPerTokenAsComplexNumbers = torch.view_as_complex(qPerTokenSplitIntoPairs)
                freqsQ = freqsCis[:qPerToken.shape[0], :].to(qPerToken.device)
                qPerTokenSplitIntoPairsRotated = torch.view_as_real(qPerTokenAsComplexNumbers * freqsQ)
                qPerTokenRotated = qPerTokenSplitIntoPairsRotated.view(qPerToken.shape)
                kPerTokenSplitIntoPairs = kPerToken.float().view(kPerToken.shape[0], -1, 2)
                kPerTokenAsComplexNumbers = torch.view_as_complex(kPerTokenSplitIntoPairs)
                freqs_k = freqsCis[:kPerToken.shape[0], :].to(kPerToken.device)
                kPerTokenSplitIntoPairsRotated = torch.view_as_real(kPerTokenAsComplexNumbers * freqs_k)
                kPerTokenRotated = kPerTokenSplitIntoPairsRotated.view(kPerToken.shape)
                qkPerToken = torch.matmul(qPerTokenRotated, kPerTokenRotated.T) / (128) ** 0.5
                mask = torch.full((seqLen, seqLen), float("-inf"), device=device)
                mask = torch.triu(mask, diagonal=1)
                qkPerTokenAfterMasking = qkPerToken + mask
                qkPerTokenAfterMaskingAfterSoftmax = torch.nn.functional.softmax(qkPerTokenAfterMasking, dim=1).to(
                    torch.bfloat16)
                qkvAttention = torch.matmul(qkPerTokenAfterMaskingAfterSoftmax, vPerToken)
                qkvAttentionStore.append(qkvAttention)
            stackedQkvAttention = torch.cat(qkvAttentionStore, dim=-1)
            wLayer = model[f"layers.{layer}.attention.wo.weight"].to(device)
            embeddingDelta = torch.matmul(stackedQkvAttention, wLayer.T)
            embeddingAfterEdit = finalEmbedding + embeddingDelta
            weightFfnNorm = model[f"layers.{layer}.ffn_norm.weight"].to(device)
            embeddingAfterEditNormalized = RmsNorm(embeddingAfterEdit, weightFfnNorm, 1e-5)
            w1 = model[f"layers.{layer}.feed_forward.w1.weight"].to(device)
            w2 = model[f"layers.{layer}.feed_forward.w2.weight"].to(device)
            w3 = model[f"layers.{layer}.feed_forward.w3.weight"].to(device)
            outputAfterFeedforward = torch.matmul(
                torch.nn.functional.silu(torch.matmul(embeddingAfterEditNormalized, w1.T)) * torch.matmul(
                    embeddingAfterEditNormalized, w3.T), w2.T)
            finalEmbedding = embeddingAfterEdit + outputAfterFeedforward

        finalEmbedding = RmsNorm(finalEmbedding, model["norm.weight"].to(device), 1e-5)
        logits = torch.matmul(finalEmbedding[-1], model["output.weight"].to(device).T)
        nextTokenScalar = torch.argmax(logits, dim=-1).item()
        tokens = torch.cat([tokens, torch.tensor([nextTokenScalar], device=device)], dim=0)
        new_char = tokenizer.decode([nextTokenScalar])
        generated_text += new_char

        display_text = "Generated Sequence: " + generated_text
        prev_length = updateLine(display_text, prev_length)

    print()
    return tokens

def precomputeFreqsCis(head_dim, seq_len, device, rope_theta):
    dim_rotary = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim_rotary, device=device).float() / dim_rotary))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def RmsNorm(tensor, normWeights, eps):
    normWeights = normWeights.to(tensor.device)
    norm = torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + eps)
    return tensor * norm * normWeights


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
    #plt.show()



def getTopNNextTokens(logits,tokenizer,N=5):
    probabilities = F.log_softmax(logits,dim=-1)
    topProbabilities, topIndices = torch.topk(probabilities,N)
    topTokens = [tokenizer.decode([idx.item()]) for idx in topIndices]
    return list(zip(topTokens,topProbabilities.tolist()))



def main():

    #Update these path based on your models.
    tokenizerPath = r"llama\checkpoints\Llama3.2-1B\tokenizer.model"
    modelPath = r"llama\checkpoints\Llama3.2-1B"
    currentOS = platform.system()

    print("\n")
    print("|---------------------|")
    print("|LLAMA3.2 FROM SCRATCH|")
    print("|_____________________|")
    print(f"Current OS:{currentOS}")

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
    word = tokenizer.decode(tokenizer.encode("hello world!\n"))
    print("\n"+word)


    #3
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device:{device}")
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
    #Benchmark prompt, it should print out 42, for the sentence below.
    #prompt = "the answer to the ultimate question of life, the universe, and everything is "
    prompt = input("Enter your prompt text: ")
    if prompt is None:
        prompt = "the answer to the ultimate question of life, the universe, and everything is "



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
    tokenEmbeddings = RmsNorm(tokenEmbeddingsUnnormalized,model["layers.0.attention_norm.weight"],1e-5)
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

    frequenciesForEachToken = torch.outer(torch.arange(len(tokens)),frequencies)
    frequenciesCis = torch.polar(torch.ones_like(frequenciesForEachToken),frequenciesForEachToken)
    frequenciesCis = torch.polar(
        torch.ones_like(frequenciesForEachToken[:, :32]),  # Only take first 32 frequencies
        frequenciesForEachToken[:, :32]
    )


    print(frequenciesCis.shape)



    value = frequenciesCis[min(3, len(tokens) - 1)]

    plt.figure()
    for i,element in enumerate(value[:len(tokens)]):
        plt.plot([0,element.real],[0,element.imag],color='blue',linewidth=1,label=f"Index: {i}")
        plt.annotate(f"{i}",xy = (element.real,element.imag),color = 'red' )

    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Plot of the one row of frequenciesCis')
    #plt.show()

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

    #WE NOW HAVE THE ATTENTION VALUE OF
    #28 Multi head attention
    qkvAttentionStore = []
    for head in range(numberOfAttentionHeads):
        qLayer0Head = qLayer0[head]
        kLayer0Head = kLayer0[head//4]
        vLayer0Head = vLayer0[head//4]

        qPerToken = torch.matmul(tokenEmbeddings,qLayer0Head.T)
        kPerToken = torch.matmul(tokenEmbeddings,kLayer0Head.T)
        vPerToken = torch.matmul(tokenEmbeddings,vLayer0Head.T)

        qPerTokenSplitIntoPairs = qPerToken.float().view(qPerToken.shape[0],-1,2)
        qPerTokenAsComplexNumbers = torch.view_as_complex(qPerTokenSplitIntoPairs)
        qPerTokenSplitIntoPairsRotated = torch.view_as_real(qPerTokenAsComplexNumbers * frequenciesCis[:len(tokens)])
        qPerTokenRotated = qPerTokenSplitIntoPairsRotated.view(qPerToken.shape)

        kPerTokenSplitIntoPairs = kPerToken.float().view(kPerToken.shape[0],-1,2)
        kPerTokenAsComplexNumbers = torch.view_as_complex(kPerTokenSplitIntoPairs)
        kPerTokenSplitIntoPairsRotated = torch.view_as_real(kPerTokenAsComplexNumbers * frequenciesCis[:len(tokens)])
        kPerTokenRotated = kPerTokenSplitIntoPairsRotated.view(kPerToken.shape)

        qkPerToken = (torch.matmul(qPerTokenRotated,kPerTokenRotated.T) / (128)**0.5).to(device)
        mask = torch.full((len(tokens), len(tokens)), float("-inf"),device = tokens.device)
        mask = torch.triu(mask,diagonal = 1)
        mask = mask.to(device)
        qkPerToken = qkPerToken.to(device)
        qkPerTokenAfterMasking = qkPerToken + mask
        qkPerTokenAfterMaskingAfterSoftmax = torch.nn.functional.softmax(qkPerTokenAfterMasking,dim=1).to(torch.bfloat16)
        qkvAttention = torch.matmul(qkPerTokenAfterMaskingAfterSoftmax,vPerToken)
        qkvAttention = torch.matmul(qkPerTokenAfterMaskingAfterSoftmax,vPerToken)
        qkvAttentionStore.append(qkvAttention)


    print("Len Of Attention Store: " + str(len(qkvAttentionStore)))


    #27
    stackedQkvAttention = torch.cat(qkvAttentionStore,dim=-1)
    print("stackedQkvAttention.shape: "+str(stackedQkvAttention.shape))

    #28
    wLayer0 = model["layers.0.attention.wo.weight"]
    print("wLayer0 shape:" + str(wLayer0.shape))

    #29
    embeddingDelta = torch.matmul(stackedQkvAttention,wLayer0.T).to(device)
    print(embeddingDelta.shape)
    tokenEmbeddingsUnnormalized = tokenEmbeddingsUnnormalized.to(device)
    embeddingDelta = embeddingDelta.to(device)
    embeddingAfterEdit = tokenEmbeddingsUnnormalized + embeddingDelta
    print(embeddingAfterEdit.shape)

    normalizedShape = model["layers.0.ffn_norm.weight"].shape
    weight = model["layers.0.ffn_norm.weight"]
    embeddingAfterEditNormalized = rms_norm(embeddingAfterEdit,normalizedShape,weight,eps=1e-5)
    print(embeddingAfterEditNormalized.shape)

    #30
    w1 = model["layers.0.feed_forward.w1.weight"]
    w2 = model["layers.0.feed_forward.w2.weight"]
    w3 = model["layers.0.feed_forward.w3.weight"]
    outputAfterFeedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embeddingAfterEditNormalized,w1.T))* torch.matmul(embeddingAfterEditNormalized,w3.T), w2.T )
    print("Output FF Shape: " + str(outputAfterFeedforward.shape))

    layer0Embedding = embeddingAfterEdit + outputAfterFeedforward
    print(layer0Embedding.shape)

    #31 Now we have to do it for 31 more layers. We can use a for loop for all layers at once.
    #Previous code block is to show, how would it be if we are to do it one layer.
    device = tokenEmbeddingsUnnormalized.device
    seq_len = len(tokens)
    dim = 2048
    n_heads = 32
    head_dim = dim // n_heads
    rope_theta = 500000.0

    freqs_cis = precomputeFreqsCis(head_dim, seq_len, device, rope_theta)
    finalEmbedding = tokenEmbeddingsUnnormalized

    for layer in range(numberOfTransformerLayers):
        qkvAttentionStore = []
        weightAttentionNorm = model[f"layers.{layer}.attention_norm.weight"]
        layerEmbeddingNorm = rms_norm(finalEmbedding, weightAttentionNorm.shape, weight=weightAttentionNorm, eps=1e-5)
        qLayer = model[f"layers.{layer}.attention.wq.weight"]
        qLayer = qLayer.view(numberOfAttentionHeads, qLayer.shape[0] // numberOfAttentionHeads, dim)
        kLayer = model[f"layers.{layer}.attention.wk.weight"]
        kLayer = kLayer.view(nKvHeads, kLayer.shape[0] // nKvHeads, dim)
        vLayer = model[f"layers.{layer}.attention.wv.weight"]
        vLayer = vLayer.view(nKvHeads, vLayer.shape[0] // nKvHeads, dim)

        for head in range(numberOfAttentionHeads):
            qLayerHead = qLayer[head]
            kLayerHead = kLayer[head // 4]
            vLayerHead = vLayer[head // 4]

            qPerToken = torch.matmul(layerEmbeddingNorm, qLayerHead.T)
            kPerToken = torch.matmul(layerEmbeddingNorm, kLayerHead.T)
            vPerToken = torch.matmul(layerEmbeddingNorm, vLayerHead.T)

            qPerTokenSplitIntoPairs = qPerToken.float().view(qPerToken.shape[0], -1, 2)
            qPerTokenAsComplexNumbers = torch.view_as_complex(qPerTokenSplitIntoPairs)
            freqs_q = freqs_cis[:qPerToken.shape[0], :].to(qPerToken.device)
            qPerTokenSplitIntoPairsRotated = torch.view_as_real(qPerTokenAsComplexNumbers * freqs_q)
            qPerTokenRotated = qPerTokenSplitIntoPairsRotated.view(qPerToken.shape)

            kPerTokenSplitIntoPairs = kPerToken.float().view(kPerToken.shape[0], -1, 2)
            kPerTokenAsComplexNumbers = torch.view_as_complex(kPerTokenSplitIntoPairs)
            freqs_k = freqs_cis[:kPerToken.shape[0], :].to(kPerToken.device)
            kPerTokenSplitIntoPairsRotated = torch.view_as_real(kPerTokenAsComplexNumbers * freqs_k)
            kPerTokenRotated = kPerTokenSplitIntoPairsRotated.view(kPerToken.shape)

            qkPerToken = torch.matmul(qPerTokenRotated, kPerTokenRotated.T) / (128) ** 0.5
            mask = torch.full((len(tokenEmbeddingsUnnormalized), len(tokenEmbeddingsUnnormalized)), float("-inf"),device=device)
            mask = torch.triu(mask, diagonal=1)
            qkPerTokenAfterMasking = qkPerToken + mask
            qkPerTokenAfterMaskingAfterSoftmax = torch.nn.functional.softmax(qkPerTokenAfterMasking, dim=1).to(torch.bfloat16)
            qkvAttention = torch.matmul(qkPerTokenAfterMaskingAfterSoftmax, vPerToken)
            qkvAttentionStore.append(qkvAttention)

        stackedQkvAttention = torch.cat(qkvAttentionStore, dim=-1)
        wLayer = model[f"layers.{layer}.attention.wo.weight"]
        embeddingDelta = torch.matmul(stackedQkvAttention, wLayer.T)
        embeddingAfterEdit = finalEmbedding + embeddingDelta
        weightFfnNorm = model[f"layers.{layer}.ffn_norm.weight"]
        embeddingAfterEditNormalized = rms_norm(embeddingAfterEdit, weightFfnNorm.shape, weight=weightFfnNorm, eps=1e-5)

        w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
        w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
        w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
        outputAfterFeedforward = torch.matmul(
            torch.nn.functional.silu(torch.matmul(embeddingAfterEditNormalized, w1.T)) *
            torch.matmul(embeddingAfterEditNormalized, w3.T), w2.T)
        finalEmbedding = embeddingAfterEdit + outputAfterFeedforward

    #32
    finalEmbedding = rms_norm(finalEmbedding, model["norm.weight"].shape, model["norm.weight"], eps=1e-5)
    print("Final Embedding Shape: " + str(finalEmbedding.shape))

    #33
    print(model["output.weight"].shape)

    #34
    logits = torch.matmul(finalEmbedding[-1],model["output.weight"].T)
    print("\nLogits Shape: " + str(logits.shape))
    generated_tokens = generateSequenceStream(model, tokens, config, tokenizer, max_new_tokens=20
                                              , device=device)
    generated_text = tokenizer.decode(generated_tokens.tolist())
    updateLine("Generated Sequence: " + generated_text)




    """
    #For next token OR 5 possible next tokens uncomment this code.
    nextToken = torch.argmax(logits,dim = -1)
    print("Next token: " + str(nextToken))
    decoded = tokenizer.decode([nextToken.item()])
    print(f"Guessed next token:{decoded}\n")
    #35 Most possible 5 outcome as next token.
    top5Result = getTopNNextTokens(logits,tokenizer,5)
    for token in top5Result:
        print(f"Token: <<[{token}]>>")
    """

if __name__ == "__main__":
    main()

