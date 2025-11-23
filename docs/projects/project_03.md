# Projeto 03 — Geração de Conteúdo Multimodal com IA Generativa

## Estudantes
- Pedro De Lucca S C Ferro
- Carlos Eduardo P. Yamada

## Resumo

Este projeto explora o uso de modelos generativos pré-treinados para geração de conteúdo multimodal (imagens e áudio) a partir de texto. Utilizando a ferramenta visual **ComfyUI** e modelos baseados em difusão (*Stable Diffusion*) e síntese de áudio, foram desenvolvidas duas pipelines: **Text-to-Image** (texto para imagem) e **Text-to-Song** (texto para música). O projeto demonstra como modelos generativos de última geração podem ser orquestrados para criar conteúdo artístico de alta qualidade através de workflows visuais.

## Objetivos

- Implementar pipelines de geração de conteúdo multimodal utilizando modelos pré-treinados
- Demonstrar o uso de *Contrastive Language–Image Pre-training* (CLIP) para condicionamento de modelos generativos
- Explorar técnicas de síntese de imagem baseadas em difusão (*diffusion models*)
- Gerar música a partir de descrições textuais e letras
- Avaliar qualitativamente a qualidade e coerência dos outputs gerados

## Metodologia

As pipelines de modelos utilizadas nesse projeto foram feitas com base em modelos previamente treinados utilizando a ferramenta visual ComfyUI[^1]

### *Text-to-image* (Texto para imagem)

A pipeline utilizada nesse exemplo é encontrada na documentação[^2].

Para exemplificar o funcionamento desse workflow, utilizaremos como base o diagrama da [Figura 1](#figure-1).

<a id="figure-1" style="text-decoration: none; color: inherit; justify-content:center;">

| ![text-to-image-workflow](../assets/projects/03/text_to_image.drawio.svg) |
| :--: |
| **Figura 1**: Fluxograma do processo de geração de imagem a partir de um texto. **Fonte**: Autor. |

</a>

Como apontado anteriormente, o modelo foi pré-treinado e seus pesos carregados em um arquivo no formato `safetensors`, que é um tipo de arquivo utilizado como alternativa aos de formato `pickle`, visto que apresenta os valores numéricos em forma de código executável[^4]. Os parâmetros do modelo são carregados devidamente em cada um dos componentes da pipeline: no CLIP, no KSampler e no VAE.

Como podemos observar, o processo inicia com o condicionamento do modelo com *Contrastive Language–Image Pre-training* (CLIP)[^3], seja ele positivo, ou seja, inserindo características que **são** desejadas na imagem a ser gerada, ou negativo, inserindo características **não** desejadas na imagem de saída.

Juntamente à entrada, que é uma imagem com ruído, os condicionamentos servem de entrada para o KSampler, que aplica o [modelo treinado](https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors?download=true) para remover o ruído da imagem no espaço latente. Dessa forma, a imagem sem ruídos é encaminhada para o decoder do *Variational Auto Encoder* (VAE), que decodifica a imagem no espaço latente para uma imagem no formato original. 

O funcionamento da ferramenta pode ser visualizado pelo [Vídeo 1](#video-1).

<a id="video-1" style="text-decoration: none; color: inherit; justify-content:center;">

| <iframe width="560" height="315" src="https://www.youtube.com/embed/8g8q5ul2rbw?si=AZb5PfP9nk11oGCn" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe> |
| :--: |
| **Vídeo 1**: Funcionamento da ferramenta de acordo com as especificações. **Fonte**: Autor. |

</a>

Além dos exemplos apresentados no vídeo, as seguintes imagens ilustradas na Figura 2 representam resultados do processo a partir do texto de entrada:

=== "Positivo"
    ```
    anime style, 1girl with long pink hair, cherry blossom background, studio ghibli aesthetic, soft lighting, intricate details

    masterpiece, best quality, 4k
    ```

=== "Negativo"
    ```
    low quality, blurry, deformed hands, extra fingers
    ``` 

<a id="figure-2" style="text-decoration: none; color: inherit; justify-content:center;">

| ![text-to-image-results](../assets/projects/03/outputs_tti.png) |
| :--: |
| **Figura 2**: Resultados do *workflow* apresentado. **Fonte**: Autor. |

</a>

### *Text-to-song* (Texto para música)

O fluxo estudado nesse exemplo pode ser encontrado no tutorial apresentado na documentação oficial[^5].

<a id="figure-3" style="text-decoration: none; color: inherit; justify-content:center;">

| ![text-to-image-results](../assets/projects/03/text_to_song.drawio.svg) |
| :--: |
| **Figura 3**: Funcionamento da ferramenta de acordo com as especificações. **Fonte**: Autor. |

</a>

Como ilustrado na [Figura 3](#figure-3), o funcionamento da *pipeline* é semelhante ao do exemplo de [Text-to-image](#text-to-image-texto-para-imagem). A mudança que podemos observar em relação ao fluxo anterior é em relação às entradas, que nesse caso, não existe um condicionamento negativo introduzido pelo CLIP. Além disso, são introduzidos outros dois parâmetros para geração do áudio: o volume da voz e o tempo total de áudio a ser produzido. O volume é um parâmetro usado em uma operação no espaço latente sobre o modelo, como pode ser observado na imagem. Com isso, o KSampler reúne os dados e aplica o modelo sobre um áudio vazio gerado com o tempo especificado e produz uma música conforme especificado pelas *tags* e letra de *input*.

Foram gerados dois áudios de acordo com as entradas especificadas abaixo.

!!! warning "Cuidado com o volume!"
    **Abaixe** o som do dispositivo ou dos áudios clicando passando o cursor por cima do botão de volume do áudio, pois eles vêm configurados com o volume no **máximo** por padrão.

!!! example Exemplo 1

    === "Letra"
        ```
        [verse]

        [zh]wo3zou3guo4shen1ye4de5jie1dao4
        [zh]leng3feng1chui1luan4si1nian4de5piao4liang4wai4tao4
        [zh]ni3de5wei1xiao4xiang4xing1guang1hen3xuan4yao4
        [zh]zhao4liang4le5wo3gu1du2de5mei3fen1mei3miao3

        [chorus]

        [verse]​
        [ko]hamkke si-kkeuleo-un sesang-ui sodong-eul pihae​
        [ko]honja ogsang-eseo dalbich-ui eolyeompus-ileul balaboda​
        [ko]niga salang-eun lideum-i ganghan eum-ag gatdago malhaess-eo​
        [ko]han ta han tamada ma-eum-ui ondoga eolmana heojeonhanji ijge hae

        [bridge]
        [es]cantar mi anhelo por ti sin ocultar
        [es]como poesía y pintura, lleno de anhelo indescifrable
        [es]tu sombra es tan terca como el viento, inborrable
        [es]persiguiéndote en vuelo, brilla como cruzar una mar de nubes

        [chorus]
        [fr]que tu sois le vent qui souffle sur ma main
        [fr]un contact chaud comme la douce pluie printanière
        [fr]que tu sois le vent qui s'entoure de mon corps
        [fr]un amour profond qui ne s'éloignera jamais
        ```

    === "*Tags*"
        ```
        synthwave, techno, synthpop, futuristic, electro, with liquid drum & bass drive.
        Restless, confident, dreamy mood at 128 BPM.
        Analog bass, pulsating arps, percussive synth stabs, gated drums.
        Quick build,  then explosive drum burst, then clean fade.
        Breathy, rhythmic female vocals, minimal emotion, metallic echo.
        ```

    ![type:audio](../assets/projects/03/sample1.mp3)

!!! example Exemplo 2

    === "Letra"
        ```
        Verse
        Neon rain on my screen,
        Dreams compile in silver sheen.
        No weight, just motion,
        I’m plugged into emotion.

        Chorus
        Comfy Cloud — breathing light,
        Code and color, spark and wire.
        Drift through data, feel alive,
        In your circuits, I arrive.
        ``` 

    === "*Tags*"
        ```
        synthwave, techno, synthpop, futuristic, electro, with liquid drum & bass drive.
        Restless, confident, dreamy mood at 128 BPM.
        Analog bass, pulsating arps, percussive synth stabs, gated drums.
        Quick build,  then explosive drum burst, then clean fade.
        Breathy, rhythmic female vocals, minimal emotion, metallic echo.
        ```

    ![type:audio](../assets/projects/03/sample2.mp3)

## Discussão

### Pontos Fortes

1. **Qualidade dos outputs**: Ambos os modelos (Text-to-Image e Text-to-Song) produziram resultados de alta qualidade, demonstrando a maturidade dos modelos generativos atuais
2. **Flexibilidade do workflow**: A arquitetura baseada em ComfyUI permite fácil experimentação e ajuste de parâmetros sem necessidade de programação extensiva
3. **Condicionamento efetivo**: O uso de CLIP para condicionamento positivo e negativo permitiu controle fino sobre as características desejadas nas imagens geradas
4. **Multimodalidade**: Demonstração bem-sucedida de geração em duas modalidades distintas (visual e auditiva) usando abordagens conceituais similares

### Limitações

1. **Dependência de modelos pré-treinados**: Não houve treinamento customizado, limitando a personalização para domínios específicos
2. **Avaliação qualitativa**: Falta de métricas quantitativas objetivas para avaliar a qualidade dos outputs (FID para imagens, MOS para áudio)
3. **Controle limitado**: Apesar do condicionamento via CLIP, o controle fino sobre detalhes específicos permanece desafiador
4. **Custo computacional**: Modelos de difusão requerem recursos significativos, especialmente para geração de alta resolução
5. **Reprodutibilidade**: Variações estocásticas inerentes aos modelos generativos podem dificultar a reprodução exata dos resultados

### Melhorias Possíveis

- **Fine-tuning**: Treinar os modelos em datasets específicos para melhorar resultados em domínios particulares
- **Métricas quantitativas**: Implementar FID (*Fréchet Inception Distance*) para imagens e MOS (*Mean Opinion Score*) para áudio
- **ControlNet**: Adicionar condicionamento espacial adicional para maior controle sobre composição de imagens
- **Pós-processamento**: Aplicar técnicas de upscaling e refinamento para melhorar qualidade final
- **Batch generation**: Gerar múltiplas variações para seleção dos melhores resultados
- **Integração de modelos mais recentes**: Explorar Stable Diffusion XL, DALL-E 3 ou Midjourney para comparação

## Conclusão

O projeto demonstrou com sucesso a aplicação de modelos generativos pré-treinados para criação de conteúdo multimodal de alta qualidade. As pipelines desenvolvidas em ComfyUI ilustram como ferramentas visuais podem democratizar o acesso a tecnologias de IA generativa, permitindo experimentação sem necessidade de expertise profunda em programação.

**Text-to-Image** produziu imagens coerentes e esteticamente agradáveis, com controle efetivo através de prompts positivos e negativos. O uso de CLIP para condicionamento e VAE para decodificação no espaço latente mostrou-se uma abordagem robusta.

**Text-to-Song** gerou composições musicais em múltiplos idiomas, demonstrando a versatilidade dos modelos generativos além da modalidade visual. A capacidade de controlar aspectos como BPM, estilo e vocalização através de tags textuais abre possibilidades criativas interessantes.

[^1]:
    [ComfyUI | Generate video, images, 3D, audio with AI](https://www.comfy.org)

[^2]:
    [ComfyUI Text to Image Workflow](https://docs.comfy.org/tutorials/basic/text-to-image)

[^3]:
    [CLIP: Connecting text and images](https://openai.com/index/clip)

[^4]:
    [Safetensors](https://huggingface.co/docs/safetensors/index)

[^5]:
    [ComfyUI ACE-Step Native Example](https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1)