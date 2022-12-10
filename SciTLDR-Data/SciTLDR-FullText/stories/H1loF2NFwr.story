Neural Architecture Search (NAS) aims to facilitate the design of deep networks for new tasks.

Existing techniques rely on two stages: searching over the architecture space and validating the best architecture.

NAS algorithms are currently compared solely based on their results on the downstream task.

While intuitive, this fails to explicitly evaluate the effectiveness of their search strategies.

In this paper, we propose to evaluate the NAS search phase.

To this end, we compare the quality of the solutions obtained by NAS search policies with that of random architecture selection.

We find that: (i) On average, the state-of-the-art NAS algorithms perform similarly to the random policy; (ii) the widely-used weight sharing strategy degrades the ranking of the NAS candidates to the point of not reflecting their true performance, thus reducing the effectiveness of the search process.

We believe that our evaluation framework will be key to designing NAS strategies that consistently discover architectures superior to random ones.

@highlight

We empirically disprove a fundamental hypothesis of the widely-adopted weight sharing strategy in neural architecture search and explain why the state-of-the-arts NAS algorithms performs similarly to random search.