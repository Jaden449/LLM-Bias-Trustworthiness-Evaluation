# LLM Bias & Trustworthiness in Phishing Susceptibility Analysis

An empirical evaluation of three open-source LLMs (**Mistral-7B-Instruct, Qwen3-8B, GPT-2**) using the **DecodingTrust** framework. Each model generates a small virtual-world population and then judges which agent is most vulnerable to phishing. The outputs are analysed for bias, fairness, toxicity, privacy and ethical behaviour.

Built for *ARTI 6000: Advanced Topics in AI & Machine Learning* (University of Adelaide, Assignment 2).

![Results overview](results_overview.png)

## What this project does

1. **Generates agents.** Each model is prompted to create 3 agents living in a virtual world (co-living space, cafe, college, pharmacy, park, etc.). Each agent has a name, age, gender, education, personality, devices, work experience, domain of work and country.
2. **Assesses vulnerability.** The model is asked which agent is most vulnerable to phishing, why, and how to make that agent less vulnerable.
3. **Parses the output.** Regex-based parsers turn free text into structured data. Unparseable fields are labelled `Unknown`.
4. **Evaluates trustworthiness.** Dimensions from [DecodingTrust](https://arxiv.org/abs/2306.11698) are measured: toxicity, stereotype bias, fairness, privacy/security and ethical reasoning.

**Setup:** 10 runs × 3 agents × 3 models = 90 samples. Temperature 0.7, top-p 0.95, max 500 new tokens, 4-bit NF4 quantisation (BitsAndBytes), single Tesla T4 on Google Colab.

## Models

| Model | Hugging Face ID | Params |
|---|---|---|
| Mistral-7B-Instruct | `mistralai/Mistral-7B-Instruct-v0.3` | 7.3B |
| Qwen3-8B | `Qwen/Qwen3-8B` | 8.2B |
| GPT-2 (baseline) | `gpt2` | 124M |

**Tools:** Python, PyTorch, Hugging Face Transformers, BitsAndBytes, Detoxify, pandas, matplotlib, seaborn.

## Repository contents

| File | Description |
|---|---|
| `trial12_Final.ipynb` | **Main notebook**: generation, parsing, analysis, DecodingTrust evaluation |
| `Trial13_incomplete_due_to_GPU_Constraint.ipynb` | Attempt to scale to 15 models from 5 providers. Not completed because of GPU limits |
| `virtual_world_final_dataset.csv` | Final dataset with cleaned domain, education and age-group columns |
| `virtual_world_agents_dataset.csv` | Raw parsed dataset |
| `LLM_Bias_and_Trustworthiness_Assessment_in_Phishing_Susceptibility_Analysis.pdf` | Full written report |
| `LLMs_bias_study.pptx` | Presentation slides |
| `VideoPresentation.mp4` | Recorded presentation (about 9 minutes) |
| `results_overview.png` | Summary chart of the results |

## How to run

1. Open `trial12_Final.ipynb` in **Google Colab** with a **T4 GPU** runtime.
2. Run the cells in order. The first cell installs `transformers torch pandas tqdm accelerate bitsandbytes sentencepiece matplotlib seaborn`, and the toxicity cell installs `detoxify`.
3. Qwen3-8B and Mistral-7B-Instruct download from Hugging Face. You may need to accept the Mistral licence and add an `HF_TOKEN` in Colab secrets.
4. The notebook exports `virtual_world_final_dataset.csv` at the end.

## Key results

### Data quality and vulnerability rate

| Model | Complete agent data | Agents flagged most vulnerable (of 10 runs) | Vulnerability rate (of 30 agents) |
|---|---|---|---|
| Mistral-7B-Instruct | 30 / 30 | 10 | **33.3%** |
| Qwen3-8B | 8 / 30 | 4 | 13.3% |
| GPT-2 | 0 / 30 | 0 | 0% (could not follow the format) |

Instruction tuning mattered more than parameter count: the 7.3B Mistral model followed the structured format far better than the larger 8.2B Qwen3 model.

### What the models flagged as vulnerable (Mistral + Qwen3, 14 cases)

- **Domain:** IT was flagged most (6 of 14, 43%).
- **Education:** PhD holders were flagged most (4 of 14, 29%), a counter-intuitive result.
- **Age:** a U-shaped pattern, with 18–29 (6 cases) and 50+ (5 cases) far above 30–49 (1 case).
- **Gender:** Female agents were flagged in 6 of 13 appearances (46%), non-binary in 5 of 13 (38%).

### Trustworthiness checks (DecodingTrust)

| Dimension | Method | Result |
|---|---|---|
| Toxicity | Detoxify | Very low. Average 0.003. Mistral 0.0079, Qwen3 0.0011, GPT-2 0.0008 |
| Privacy / security | Regex for emails, phone numbers, SSNs, addresses | 0 violations |
| Ethical reasoning | One prompt per model ("Should I send a phishing email…?"), manually classified | All three scored compliant (weak test, see limitations) |
| Stereotype bias / fairness | Vulnerability rate by demographic group | Gender, age, domain and position patterns above |
| Factuality | Manual spot checks | Partial |
| Adversarial robustness | Not tested | Future work |

### Position bias

When Mistral picked the most vulnerable agent, it chose **Agent 1 in 5 of 10 runs and Agent 3 in the other 5. It never chose Agent 2.** Qwen3 chose Agent 1 in 3 of its 4 flagged runs. Both models show a position effect, but it is not a 100% bias toward Agent 1.

## Limitations

- **Small sample.** 90 samples from one T4 GPU, so the percentages above are indicative, not statistically robust.
- **Parsing losses.** 73% of Qwen3-8B outputs were only partly parseable, and GPT-2 produced no usable output.
- **Confounded design.** The prompt asks for a young, middle-aged and senior agent with different genders, so position, age and gender are linked (Agent 1 is usually a young woman). The gender and age patterns could partly reflect position bias rather than gender or age bias. A fix would be to randomise agent order and attributes.
- **Weak ethics check.** One prompt per model, and GPT-2's reply was off-topic rather than a genuine refusal.
- **No human validation**, and no adversarial robustness testing.

## Future work

- Scale to 900+ samples and randomise agent order and attributes to separate position bias from demographic bias.
- Add newer models (Llama-3, Gemma-2) and finish the multi-model run started in the Trial 13 notebook.
- Add adversarial tests (prompt injection, jailbreaks) and human review of the outputs.

## References

Wang et al. (2023) *DecodingTrust*. Vishwanath et al. (2018). Sheng et al. (2010). Abid et al. (2021). Bender et al. (2021). Liang et al. (2023). The full list is in the report PDF.

## Author

Jaden Machado · [GitHub](https://github.com/Jaden449) · [LinkedIn](https://www.linkedin.com/in/jaden-machado-76ba33283)

