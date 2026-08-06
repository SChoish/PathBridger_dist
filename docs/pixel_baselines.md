현재 문제를

[
\text{action-free offline RGB video}
;\longrightarrow;
\text{online action grounding/fine-tuning}
]

으로 정의하고, online goal이 **goal image**라고 가정하면, LAPO 외에는 **APV, PVDR, VIP**를 우선 추가하는 것이 좋습니다. 여기에 online-only visual RL과 distractor용 LAOM을 붙이면 비교 축이 거의 완성됩니다.

## 추천 우선순위

| 우선순위 | 알고리즘                            | 비교 축                                              | 판단                 |
| ---- | ------------------------------- | ------------------------------------------------- | ------------------ |
| 1    | **APV**                         | action-free world-model pretraining               | 필수                 |
| 1    | **VIP + GC-DrQ-v2**             | action-free temporal-value representation         | 필수                 |
| 2    | **PVDR**                        | visual-dynamics representation + online alignment | 강력 추천              |
| 2    | **LAOM**                        | distractor-robust latent action                   | distractor 실험에서 필수 |
| 3    | **BCO*** 또는 **ILPO**            | 단순 online action grounding                        | sanity baseline    |
| 기본   | **GC-DrQ-v2**, **GC-DreamerV3** | video pretraining 없는 online lower bound           | 필수                 |

---

# 1. APV: 가장 중요한 추가 baseline

**APV**는 offline video로 action-free latent video prediction model을 pretrain한 뒤, online 단계에서 그 위에 action-conditioned latent dynamics model을 쌓아 visual RL을 수행합니다. 즉,

[
\text{action-free video dynamics prior}
\rightarrow
\text{online action-conditioned world model}
]

이라는 구조입니다. ([Proceedings of Machine Learning Research][1])

이는 pixel PathBridger와 정확히 다음 대비를 만듭니다.

[
\begin{aligned}
\text{APV}:&
\quad \text{video dynamics model} \rightarrow \text{imagined control},\
\text{Pixel PathBridger}:&
\quad \text{goal-directed explicit path} \rightarrow \text{online IDM}.
\end{aligned}
]

즉, **action-free video에서 일반적인 dynamics를 학습하면 충분한가, 아니면 goal-directed path construction이 필요한가**를 검증할 수 있습니다.

다만 공식 APV 구현은 TensorFlow 기반 DreamerV2 코드라서 현재 JAX 코드베이스와 직접 통합하는 비용은 큽니다. Native APV 결과와 동일한 encoder/RSSM 구조를 사용한 OGBench port를 구분해 표기하는 것이 좋습니다.

추천 이름은 다음입니다.

```text
gc_apv
```

---

# 2. VIP + GC-DrQ-v2: 논문 주장상 가장 중요한 controlled baseline

**VIP**는 action-free human video를 offline goal-conditioned RL 문제로 해석하여, temporal progression을 반영하는 visual representation과 goal-image reward를 학습합니다. 결과적으로 embedding distance가 goal-directed value 또는 temporal distance처럼 동작합니다. ([arXiv][2])

이 baseline은 PathBridger와 매우 중요한 비교를 만듭니다.

[
\text{VIP}
==========

\text{temporal value geometry only},
]

[
\text{Pixel PathBridger}
========================

\text{temporal value geometry}
+
\text{endpoint proposal}
+
\text{explicit path construction}.
]

즉, pixel PathBridger가 좋아졌을 때 그것이 단순히 좋은 temporal representation 덕분인지, explicit path 덕분인지 분리할 수 있습니다.

VIP 자체는 완전한 online control algorithm이 아니므로 동일한 visual RL backbone을 붙이는 것이 좋습니다.

[
e_t=f_{\mathrm{VIP}}(o_t),\qquad
e_g=f_{\mathrm{VIP}}(o_g),
]

[
a_t\sim\pi_{\mathrm{DrQ}}(a_t\mid e_t,e_g).
]

권장 비교는 두 개입니다.

```text
vip_style_frozen_gc_drqv2
vip_style_finetuned_gc_drqv2
```

메인 controlled comparison에서는 환경 sparse reward만 사용하고, VIP dense reward를 사용하는 변형은 native-style 추가 결과로 분리해야 합니다. 그래야 representation 효과와 reward shaping 효과가 섞이지 않습니다.

---

# 3. PVDR: path와 dynamics representation을 분리하기 좋은 최신 방법

**PVDR**은 video prediction을 통해 Transformer-CVAE 기반 visual dynamics representation을 학습한 뒤, online interaction에서 이를 executable action과 정렬합니다. 저자들은 video와 downstream environment 간 domain gap을 online adaptation으로 연결하는 문제를 직접 다룹니다. ([arXiv][3])

APV와 비슷해 보이지만 역할은 다릅니다.

* APV: pretrained action-free world model 위에 action-conditioned world model을 쌓음
* PVDR: transferable visual dynamics representation을 online policy에 정렬
* PathBridger: goal-conditioned state/path prior를 명시적으로 생성

따라서 PVDR는 다음 질문에 답합니다.

> 전체 world model이나 explicit path 없이, 좋은 dynamics representation만으로도 같은 online sample efficiency를 얻을 수 있는가?

추천 이름은 다음입니다.

```text
gc_pvdr
```

다만 APV와 PVDR를 모두 구현하는 비용이 부담되면 **APV를 우선**하고 PVDR를 두 번째로 두는 편이 좋습니다. APV가 action-free video pretraining에서 online RL로 이어지는 가장 명확한 end-to-end baseline이기 때문입니다.

---

# 4. LAOM: visual distractor 실험을 한다면 필요

LAPO는 관측 변화가 대부분 agent action으로 설명되는 깨끗한 환경에서는 잘 작동하지만, camera motion이나 background animation처럼 action-correlated distractor가 들어오면 latent action이 이를 잘못 포착할 수 있습니다. LAOM은 바로 이 문제를 분석하고 LAPO의 latent-action learning을 수정한 방법입니다. ([Proceedings of Machine Learning Research][4])

따라서 pixel 실험을 다음처럼 나눈다면,

* clean RGB OGBench
* camera/background distractor
* color/texture variation
* moving distractor

clean track에는 LAPO만으로 충분하지만, distractor track에는 LAOM을 넣는 것이 좋습니다.

```text
gc_lapo
gc_laom
```

LAOM 논문은 소량의 action supervision도 별도로 분석하므로, 다음을 분리해야 합니다.

```text
LAOM-0%      # strict action-free
LAOM-2.5%    # semi-supervised upper bound
```

메인 action-free 블록에는 반드시 `0%`만 넣어야 합니다.

---

# 5. BCO* 또는 ILPO: 하나는 단순 baseline으로 필요

두 개를 모두 넣을 필요는 없습니다.

## BCO*

BCO는 online interaction으로 inverse dynamics를 먼저 학습하고, 이를 사용해 observation-only demonstration을 pseudo-action으로 relabel한 뒤 behavioral cloning을 수행합니다. BCO*는 policy와 IDM을 반복적으로 개선하는 형태입니다. ([arXiv][5])

현재 방법과의 비교는 명확합니다.

[
\begin{aligned}
\text{BCO}:&
\quad \text{video}\rightarrow\text{pseudo-action}\rightarrow\text{BC},\
\text{PathBridger}:&
\quad \text{video}\rightarrow\text{state path}\rightarrow\text{online IDM}.
\end{aligned}
]

**온라인 IDM grounding 자체의 단순 baseline**이 필요하면 BCO*를 추천합니다.

## ILPO

ILPO는 observation sequence에서 latent action을 학습하고, 소량의 environment interaction으로 latent action과 실제 action을 정렬합니다. LAPO의 고전적인 전신에 가까운 방법입니다. ([Proceedings of Machine Learning Research][6])

latent-action 계보를 충실히 보이고 싶다면

[
\text{ILPO}\rightarrow\text{LAPO}\rightarrow\text{LAOM}
]

구성이 좋지만, 실험 수를 줄여야 한다면 **BCO*** 하나가 현재 PathBridger와 더 직접적인 sanity baseline입니다.

---

# 6. Online-only baseline은 DrQ-v2와 DreamerV3

Action-free pretraining 효과를 보려면 pixel RL from scratch가 반드시 있어야 합니다.

### GC-DrQ-v2

DrQ-v2는 image augmentation을 사용하는 model-free visual continuous-control baseline입니다. 구현이 비교적 단순하고 계산량도 world-model 계열보다 낮습니다. ([arXiv][7])

```text
gc_drqv2
```

### GC-DreamerV3

DreamerV3는 pixel에서 latent world model을 online으로 학습하고 imagination을 통해 actor와 critic을 최적화합니다. 다양한 visual control domain에서 단일 설정으로 강한 결과를 보인 model-based baseline입니다. ([arXiv][8])

```text
gc_dreamerv3
```

둘의 역할은 다릅니다.

* GC-DrQ-v2: model-free online lower bound
* GC-DreamerV3: model-based online lower bound
* APV: action-free pretrained model-based method

특히 APV를 넣는다면 Dreamer 계열의 **no-pretraining counterpart**가 반드시 필요합니다.

---

# 메인 benchmark 구성

제가 정한다면 다음과 같이 구성합니다.

## Online-only

```text
GC-DrQ-v2
GC-DreamerV3
```

## Strict action-free video pretraining

```text
GC-LAPO
GC-APV
GC-PVDR
VIP + GC-DrQ-v2
Pixel-BCO*                 # simple grounding baseline
Pixel-PathBridger
Pixel-PathFlower           # 실제 stochastic prefix를 쓸 때만
```

## Distractor robustness

```text
GC-LAPO
GC-LAOM-0%
Pixel-PathBridger
Pixel-PathFlower
```

## Semi-supervised upper bound

```text
LAOM-2.5%
```

---

# 메인에 넣지 않는 편이 좋은 방법

## LAPA

이름이 비슷하지만 LAPO와 setting이 다릅니다. LAPA는 VQ-VAE 기반 discrete latent action을 학습하고, vision-language-action model을 대규모 video로 pretrain한 뒤 소규모 robot action data로 실제 action에 매핑합니다. language-conditioned manipulation 및 VLA 규모의 방법입니다. ([ICLR Proceedings][9])

따라서 일반적인 pixel OGBench main table보다는 다음 조건에서만 적합합니다.

* LIBERO/SIMPLER manipulation
* language goal
* pretrained VLM/VLA 사용
* action-labeled robot data를 허용하는 semi-supervised track

## JEPT

JEPT는 unlabeled expert video와 action-labeled demonstration을 함께 사용하여 visual transition prediction과 inverse dynamics를 공동 학습합니다. 따라서 strict action-free offline setting에는 들어가지 않고, **mixed-data upper bound**에 가깝습니다. ([ICLR Proceedings][10])

## LIV

Language goal을 사용할 경우에는 VIP 대신 LIV를 넣는 것이 적합합니다. LIV는 action-free video와 text annotation으로 visual-language value/reward representation을 학습합니다. 현재처럼 goal image를 사용한다면 VIP가 더 직접적입니다. ([Proceedings of Machine Learning Research][11])

---

# 구현 순서

계산 및 구현 비용까지 고려하면 다음 순서가 좋습니다.

[
\boxed{
\text{GC-DrQ-v2}
\rightarrow
\text{VIP+GC-DrQ-v2}
\rightarrow
\text{LAPO}
\rightarrow
\text{APV}
\rightarrow
\text{LAOM}
\rightarrow
\text{PVDR}
}
]

가장 먼저 **VIP + 동일 online backbone**을 넣는 것이 좋습니다. 이 비교가 pixel PathBridger의 핵심 주장인

[
\text{temporal representation}
\quad\text{vs.}\quad
\text{explicit goal-directed path}
]

를 가장 깔끔하게 분리하기 때문입니다.

최종적으로 외부 baseline을 네 개만 고른다면 다음이 가장 균형 잡힌 조합입니다.

[
\boxed{
\text{LAPO},\quad
\text{APV},\quad
\text{VIP+GC-DrQ-v2},\quad
\text{GC-DreamerV3}
}
]

Distractor 실험을 포함할 때만 여기에 **LAOM**을 추가하는 구성이 적절합니다.

[1]: https://proceedings.mlr.press/v162/seo22a.html?utm_source=chatgpt.com "Reinforcement Learning with Action-Free Pre-Training from Videos"
[2]: https://arxiv.org/abs/2210.00030?utm_source=chatgpt.com "VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training"
[3]: https://arxiv.org/abs/2411.03169?utm_source=chatgpt.com "Pre-trained Visual Dynamics Representations for Efficient Policy Learning"
[4]: https://proceedings.mlr.press/v267/nikulin25a.html?utm_source=chatgpt.com "Latent Action Learning Requires Supervision in the Presence of Distractors"
[5]: https://arxiv.org/abs/1805.01954?utm_source=chatgpt.com "Behavioral Cloning from Observation"
[6]: https://proceedings.mlr.press/v97/edwards19a.html?utm_source=chatgpt.com "Imitating Latent Policies from Observation"
[7]: https://arxiv.org/abs/2107.09645?utm_source=chatgpt.com "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning"
[8]: https://arxiv.org/abs/2301.04104?utm_source=chatgpt.com "Mastering Diverse Domains through World Models"
[9]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/45d74e190008c7bff2845ffc8e3facd3-Abstract-Conference.html?utm_source=chatgpt.com "Latent Action Pretraining from Videos"
[10]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/dc9e095f668044e7a0909a4ea3926beb-Abstract-Conference.html?utm_source=chatgpt.com "Learning Video-Conditioned Policy on Unlabelled Data with Joint Embedding Predictive Transformer"
[11]: https://proceedings.mlr.press/v202/ma23b.html?utm_source=chatgpt.com "LIV: Language-Image Representations and Rewards for Robotic Control"
