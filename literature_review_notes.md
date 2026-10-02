# Literature Review Notes

## 1. POMDP planning — Silver and Veness (2010)
- POMCP uses Monte Carlo tree search for online planning in large POMDPs.
- Important because I-NTMCP builds on Monte Carlo planning ideas.
- Key idea: approximate belief/planning through sampling instead of enumerating the full belief space.
- Relevance to my project: provides the planning foundation for reasoning under partial observability.

## 2. Interactive POMDPs — Gmytrasiewicz and Doshi (2005)
- I-POMDPs extend POMDPs by including models of other agents.
- Allows nested beliefs such as "I believe what the opponent believes".
- Main limitation: recursive modelling becomes computationally expensive.
- Relevance: theoretical basis for nested reasoning levels.

## 3. Approximate I-POMDP reasoning — Doshi and Gmytrasiewicz (2005)
- Uses particle filtering to approximate interactive beliefs.
- Motivation: exact I-POMDP belief representation becomes too large.
- Relevance: supports the need for approximation methods in multi-agent partial observability.

## 4. I-NTMCP — Schwartz, Zhou and Kurniawati (2022)
- Main paper underlying this project.
- Uses nested Monte Carlo tree search for online planning in Interactive-POMDPs.
- Finite nesting levels approximate recursive reasoning.
- Relevance: my experiments directly test how useful different I-NTMCP nesting levels are in pursuit-evasion.

## 5. Practical / bounded nested reasoning — Hoang and Low (2013)
- I-POMDP Lite reduces the cost of interactive reasoning.
- Shows the broader trade-off between reasoning complexity and practical performance.
- Relevance: connects to my finding that deeper nesting does not consistently improve performance.

## 6. Opponent type uncertainty — Schwartz, Kurniawati and Hutter
- Studies planning when the opponent may have different behavioural types.
- Relevance: closely related to my MIXED evader condition, where the evader policy is not known in advance.

## 7. Research gap
- Existing work develops methods for nested reasoning and opponent modelling.
- My project asks when deeper reasoning actually improves performance.
- Main focus:
  - environment structure
  - fixed vs mixed evader behaviour
  - whether nesting beyond l=1 provides additional benefit