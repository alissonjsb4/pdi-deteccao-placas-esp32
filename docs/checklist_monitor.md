# Checklist de revisão — padrões de cobrança do monitor (TI0147)

Extraído dos comentários do monitor em um trabalho da disciplina (02/07/2026).
Usar como revisão final de relatório e slides antes de entregar.

1. **Todo conceito novo tem imagem de exemplo** (CLAHE, blur, Otsu, espaços de
   cor, arquitetura…). Conceito citado sem figura = comentário garantido.
2. **Nenhuma tabela solta**: cada tabela é explicada em texto corrido — o que é
   cada linha/experimento e o que se conclui dela.
3. **Antes/depois em toda etapa de processamento** (pré-processamento e
   segmentação passo a passo).
4. **Otsu exige o histograma**: mostrar o histograma analisado com o limiar
   marcado, além do antes/depois da segmentação.
5. **Justificar o espaço de cor**: em PDI usa-se tons de cinza (1 canal); se a
   rede exige 3 canais, explicar a solução (ex.: luminância replicada — gray3)
   como forma de contornar a limitação da arquitetura.
6. **Conferir afirmações de pré-treino**: dizer explicitamente a base
   (ImageNet, no nosso caso — `weights="imagenet"` no notebook). O monitor
   verifica (cobrou COCO no outro trabalho).
7. **Nunca deixar valor sem nome** ("E qual é?"): função de perda, limiares e
   hiperparâmetros têm de ser nomeados com fórmula/valores.
8. **Tamanho de entrada afeta tempo de treino** — declarar resolução e ser
   consistente em todo o texto.
9. **Termos explicados na primeira ocorrência** (BlackHat, GIoU, PSRAM…).
10. **Experimento ruim → refazer**, não relatar com ressalvas.
11. **Vários exemplos visuais de segmentação**, não um único caso.
12. **Escolhas defendidas com trade-off** (ex.: edge × nuvem; INT8 × float).
