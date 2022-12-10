We demonstrate a low effort method that unsupervisedly constructs task-optimized embeddings from existing word embeddings to gain performance on a supervised end-task.

This avoids additional labeling or building more complex model architectures by instead providing specialized embeddings better fit for the end-task(s).

Furthermore, the method can be used to roughly estimate whether a specific kind of end-task(s) can be learned form, or is represented in, a given unlabeled dataset, e.g. using publicly available probing tasks.

We evaluate our method for diverse word embedding probing tasks and by size of embedding training corpus -- i.e. to explore its use in reduced (pretraining-resource) settings.

Unsupervisedly pretrained word embeddings provide a low-effort, high pay-off way to improve the performance of a specific supervised end-task by exploiting Transfer learning from an unsupervised to the supervised task.

Additionally, recent works indicate that universally best embeddings are not yet possible, and that instead embeddings need to be tuned to fit specific end-tasks using inductive bias -i.e.

semantic supervision for the unsupervised embedding learning process BID1 BID13 .

This way, embeddings can be tuned to fit a specific Single-task (ST) or Multi-task (MT: set of tasks) semantic BID16 BID7 .

Hence the established notion, that in order to fine-tune embeddings for specific end-tasks, labels for those endtasks a required.

However, in practice, especially in industry applications, labeled dataset are often either too small, not available or of low quality and creating or extending them is costly and slow.

Instead, to lessen the need for complex supervised (Multi-task) fine-tuning, we explore using unsupervised fine-tuning of word embeddings for either a specific end-task (ST) or a set of desired end-tasks (MT).

By taking pretrained word embeddings and unsupervisedly postprocessing (finetuning) them, we evaluate postprocessing performance changes on publicly available probing tasks developed by BID6 1 to demonstrate that widely used word embeddings like Fasttext and GloVe can either: (a) be unsupervisedly specialized to better fit a single supervised task or, (b) can generally improve embeddings for multiple supervised end-tasks -i.e.

the method can optimize for single and Multi-task settings.

As in standard methodology, optimal postprocessed embeddings can be selected using multiple proxy-tasks for overall improvement or using a single endtask's development split -e.g.

on a fast baseline model for further time reduction.

Since most embeddings are pretrained on large corpora, we also investigate whether our method -dubbed MORTY -benefits embeddings trained on smaller corpora to gauge usefulness for low-labeling-resource domains like biology or medicine.

We demonstrate the method's application for Single-task, Multitask, small and large corpus-size setting in the evaluation section 3.

Finally, MORTY (sec. 2), uses very little resources 2 , especially regarding recent approaches that exploit unsupervised pretaining to boost end-task performance by adding complex pretraining components like ELMo, BERT BID14 BID2 which may not yet be broadly usable due to their hardware and processing time requirements.

As a result, we demonstrate a simple method, that allows further pretraining exploitation, while requiring minimum extra effort, time and compute resources.

We unsupervisedly create specialized inputs for supervised end-tasks using Multiple Ordinary Reconstructing Transformations to Ymprove embedding 3 of the original inputs.

Specifically, as seen in the FIG0 , MORTY uses multiple, separate Autoencoders that create new representations by learning to reconstruct the original pretrained embeddings 4 .

The resulting representations (postprocessed embeddings) can provide both: (a) better performance for a single supervised probing task (ST), and (b) boost performance of multiple tasks (MT) or overall performance across all probing tasks.

To pick an optimal MORTY embedding for single and Multi-task settings, we can either use proxy-tasks or an end-task(s)'s development split(s).

In practice, MORTY can be efficiently trained as a (data) hyperparameter to the end or proxy tasks -see details in section 3.

Embeddings by corpus sizes: We train 100 dimensional embeddings with Fasttext BID0 5 and GloVe BID12 6 on wikitext-2 and wikitext-103 created by BID10 .

By also using public Fasttext and GloVe 7 embeddings we can 3 Y since labels/outputs (embeddings) are reconstructed 4 Link to code will be made public after publication.

BID6 .

It is split into three semantic categories: (a) word similarity (6 tasks), (b) word analogy (3 tasks), and (c) word and sentence classification/ categorization (9 tasks).

In the following we evaluate embedding performance scores for Fasttext and GloVe and their percentual change after postprocessing with MORTY.

We evaluate MORTY for Single-task (ST) and Multi-task (MT) application optimization.

Results can be seen in Tables 1 and 2 .

For the Singletask application setting we show MORTY's percentual performance impact in the ST % change column, where results are produced by choosing the best MORTY embedding per individual task -18 MORTYs.

For the Multi-task application setting the MT % change column shows percentual performance impact when choosing the MORTY with the best over-all-tasks score Σ. Finally, we evaluate by smaller (wikitext 2M, 103M) and very large (600B/840B) training corpus sizes, as well as by the three semantic property categories described in section 2.

Model performances, given in Tables 1 and 2 , are 5-run averages of Fasttext and GloVe per corpus sizes 2M and 103M, while the public 600B/ 840B were evaluated once.

MORTY's performances on 2M and 103M are given as relative, percentual change, averaged over 5 according base-embedder runs.

For Fasttext and GloVe -run 5 times on 2M and 103M -we can see in each table's left column that 8 Fasttext was trained using the implementation's (fasttext.cc) default parameters.

GloVe was trained with the same parameters as in BID12 ) - Figure 4b .

Though, 4a gave the same results.9 < 0.5% between runs for both Fasttext and GloVe .

results for classification (category), similarity and analogy improve expectedly with corpora size.

When looking at the middle columns (MT % change) we see that using a single best MORTY improves overall performance Σ 10 -the sum of 18 tasks -by roughly 2 − 9% compared to base embeddings, especially for smaller corpus sizes.

While Fasttext benefits more than GloVe from MORTY, both perform particularly badly for analogy tasks on the smaller corpora 2M and 103M where Fasttext beats GloVe, especially after applying MORTY.

This is also reflected in the small/medium set Google and MSR analogy scores doubling and tripling (still middle column).

However, public GloVe (840B) has the Table 2 : MORTY on Glove: Same as in Table 1 but for GloVe .

best analogy performance, while MORTY further improves analogy scores for both public embeddings -600B/840B.

Additionally, for similarity we see decent improvements for the smallest corpus, but not for larger corpora as base Fasttext already has higher performance.

Classification exhibits more mixed, smaller, changes.

For smaller datasets Fasttext clearly beats GloVe in overall performance (8.83 vs. 6.68).

For public embeddings (600B/840B) base scores are equal.

GloVe leads analogy.

Fasttext leads similarity and improves more from MORTY.

However, despite GloVe's significantly lower base performance on smaller datasets, MORTY used on GloVe produces lower but more stable improvements for the MT setting (middle column).

Generally, we see both performance increases and drops for individual task, especially on the smaller datasets and for Fasttext, indicating that, an overall best MORTY specializes the original Fasttext embedding to better fit a specific subset of the 18 tasks, while still being able to beat base embeddings in overall (Σ) score.

MORTY for Single-task application: When looking at the ST % change columns in both tables we see Single-task (ST) results for 18 individually best MORTY embeddings.

Both Fasttext and GloVe show consistent improvements from using MORTY, with Fasttext exhibiting more improvement potential on smaller datasets, while GloVe shows more ST improvement on very large datasets, indicating that MORTY benefits both embedding methods.

Particularly when base scores for a task were low -e.g.

for the analogy tasks -MORTY often improved upon the particular baseembedding's weaknesses.

Low-resource benefits: MORTY seems especially beneficial on the smaller corpora (2M and 103M) for both MT and ST applications as well as for Fasttext and GloVe -indicating that MORTY is well suited for low-resource settings.

MORTY training: Finally, we found optimal parameters for training MORTY to be close to or the same as the original embedding model -i.e.

same learning rate, embedding size and epochs.

Though we initially experimented with variations such as sparse and denoising, or sigmoid and ReLu activations, we found linear activation, (over)complete Autoencoders trained with bMSE (batch-wise mean squared error) to perform best.

In settings, where no supervised, or proxy dataset(s) are available to select the best MORTY embedding we found a practical setting for Fasttext and GloVe that consistently increased overall probing-task performance by simply training with a learning rate lr = 0.01 11 , for 1 epoch, and a representation size equal to or twice as large as the original embedding -i.e.

train an (over)complete representation.

When compressing from the original embedding size, e.g. from 100 to 20, space reduction outweighed performance loss -so larger vocabularies are usable at sublinear performance loss 12 .

More involved parameter exploration yielded little extra gains.

Methods of information transfer from or to supervised tasks has been heavily focused in recent Transfer Learning literature, while transfer between unsupervised tasks received less attention.

Unsupervised-to-Supervised: For word meaning transfer, Word2Vec BID11 , Fastext BID0 BID12 and GloVe BID12 provide unsupervisedly pretrained embeddings that can be used to generally improve performance on arbitrary supervised end-tasks.

Supervised-tounsupervised: However, transfer can also be used vice versa, to (learn to) specialize embeddings to better fit a specific supervised signal BID17 BID15 or even to enforce that generally relevant semantics are encoded by using auxiliary Multi-task supervision BID8 BID4 .

The approach by BID15 is especially interesting since they proposed an automated method (Bayesian optimization) for tuning embeddings to a specific end-task.

Supervised-to-supervised: Another way to realize knowledge transfer is between supervised tasks, that can be exploited successively BID9 , jointly BID8 and in joint-succession BID5 to improve each others performance.

Unsupervised-to-unsupervised: More recently, BID3 proposed a GloVe modification that retrofits publicly available (external) GloVe embeddings to produce better domain embeddings for a specific end-task.

In contrast, MORTY does not require external (public) embeddings, does not require target domain texts 13 , can be applied to embeddings produced by any embedding method, and can be used with or without direct supervision by a desired (set of) end-tasks -resulting in low-effort usage.

MORTY instead uses unsupervised fine-tuning of embeddings to better fit one or more desired supervised semantics.

This way, we can avoid manual extensions like complex multitask learning setups or creating potentially hard to come by taskrelated supervised data sets.

Instead MORTY can be optimized as a data-input parameter for a desired (set of) end-tasks or proxy-tasks (proxysemantics), and shows additional benefits in lowresource settings.

We demonstrated a low-effort method to unsupervisedly construct task-optimized word embeddings from existing ones to gain performance on a (set of) supervised end-task(s).

Despite its simplicity, MORTY is able to produces significant performance improvements for Single and Multi-task supervision settings as well as for a variety of desirable word encoding properties -even on smaller corpus sizes -while forgoing additional labeling or building more complex model architectures.

<|TLDR|>

@highlight

Morty refits pretrained word embeddings to either: (a) improve overall embedding performance (for Multi-task settings) or improve Single-task performance, while requiring only minimal effort.