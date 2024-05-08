<h1 align="center"> 
Region-Disentangled Diffusion Model for High-Fidelity PPG-to-ECG Translation
</h1>

<h3 align="center">
AAAI 2024
</h3>
<h3 align="center">
<a href="https://www.debadityashome.com">Debaditya Shome</a>
&nbsp;
Pritam Sarkar
&nbsp;
Ali Etemad
</h3>


### Updates
- [x] Paper
- [x] RDDM algorithm
- [x] Training code
- [x] Pretrained diffusion weights (https://drive.google.com/drive/folders/1Z7JQ5VdTrekx4lbARJNIUiR5D-4Kz7wg?usp=sharing)
- [ ] CardioBench
- [x] Evaluation code

#### ** You may follow this repo to receive future updates. **


### Abstract
The high prevalence of cardiovascular diseases (CVDs) calls for accessible and cost-effective continuous cardiac monitoring tools. Despite Electrocardiography (ECG) being the gold standard, continuous monitoring remains a challenge, leading to the exploration of Photoplethysmography (PPG), a promising but more basic alternative available in consumer wearables. This notion has recently spurred interest in translating PPG to ECG signals. In this work, we introduce Region-Disentangled Diffusion Model (RDDM), a novel diffusion model designed to capture the complex temporal dynamics of ECG. Traditional Diffusion models like Denoising Diffusion Probabilistic Models (DDPM) face challenges in capturing such nuances due to the indiscriminate noise addition process across the entire signal. Our proposed RDDM overcomes such limitations by incorporating a novel forward process that selectively adds noise to specific regions of interest (ROI) such as QRS complex in ECG signals, and a reverse process that disentangles the denoising of ROI and non-ROI regions. Quantitative experiments demonstrate that RDDM can generate high-fidelity ECG from PPG in as few as 10 diffusion steps, making it highly effective and computationally efficient. Additionally, to rigorously validate the usefulness of the generated ECG signals, we introduce CardioBench, a comprehensive evaluation benchmark for a variety of cardiac-related tasks including heart rate and blood pressure estimation, stress classification, and the detection of atrial fibrillation and diabetes. Our thorough experiments show that RDDM achieves state-of-the-art performance on CardioBench. To the best of our knowledge, RDDM is the first diffusion model for cross-modal signal-to-signal translation in the bio-signal domain.


### Citation
If you find this repository useful, please consider giving a star :star: and citation using the given BibTeX entry:

```
@article{shome2023region,
  title={Region-Disentangled Diffusion Model for High-Fidelity PPG-to-ECG Translation},
  author={Shome, Debaditya and Sarkar, Pritam and Etemad, Ali},
  journal={arXiv preprint arXiv:2308.13568},
  year={2023}
}
```

### Acknowledgments
This work was supported by Mitacs, Vector Institute, and
Ingenuity Labs Research Institute.
