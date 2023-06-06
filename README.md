# MeZO: Fine-Tuning Language Models with Just Forward Passes

This is the implementation for the paper [Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/pdf/2305.17333.pdf). 
In this paper we propose a memory-efficient zeroth-order optimizer (**MeZO**),
adapting the classical zeroth-order SGD method to operate in-place, thereby fine-tuning language models (LMs) with the same memory footprint as inference.

With a single A100 80GB GPU, MeZO can train a 30-billion parameter OPT model, whereas fine-tuning with Adam can train only a 2.7B LM.
MeZO demonstrates comparable performance to fine-tuning with backpropagation across multiple tasks, with up to 12× memory reduction. MeZO is also compatible with both full-parameter and parameter-efficient tuning techniques such as LoRA and prefix tuning. We also show that MeZO can effectively optimize non-differentiable objectives (e.g., maximizing accuracy or F1).


<p>
  <img src="https://github.com/princeton-nlp/MeZO/blob/main/assets/fig2.png?raw=true" alt="Fig" width="100%"/>
  <em>
  GPU memory usage comparison between zero-shot, in-context learning (ICL), Adam fine-tuning (FT), and our proposed MeZO.
  </em>
</p>

<p>
  <img src="https://github.com/princeton-nlp/MeZO/blob/main/assets/fig1.png?raw=true" alt="Fig" width="100%"/>
  <em>
  OPT-13B results with zero-shot, in-context learning (ICL), MeZO (we report the best among MeZO/MeZO (LoRA)/MeZO (prefix)), and fine-tuning with Adam (FT). MeZO demonstrates superior results over zero-shot and ICL and performs on par with FT (within 1%) on 7 out of 11 tasks, despite using only 1/12 memory.
  </em>
</p>


## Outline

For reproducing RoBERTa-large experiments, please refer to the `medium_models` folder. For OPT experiments, please refer to the `large_models` folder.


## Citation

```bibtex
@article{malladi2023mezo,
   title={Fine-Tuning Large Language Models with Just Forward Passes},
   author={Malladi, Sadhika and Gao, Tianyu and Nichani, Eshaan and Damian, Alex and Lee, Jason D and Chen, Danqi and Arora, Sanjeev},
   year={2023}
}
```
