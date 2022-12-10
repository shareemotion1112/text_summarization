Generative adversarial networks (GANs) train implicit generative models through solving minimax problems.

Such minimax problems are known as nonconvex- nonconcave, for which the dynamics of first-order methods are not well understood.

In this paper, we consider GANs in the type of the integral probability metrics (IPMs) with the generator represented by an overparametrized neural network.

When the discriminator is solved to approximate optimality in each iteration, we prove that stochastic gradient descent on a regularized IPM objective converges globally to a stationary point with a sublinear rate.

Moreover, we prove that when the width of the generator network is sufficiently large and the discriminator function class has enough discriminative ability, the obtained stationary point corresponds to a generator that yields a distribution that is close to the distribution of the observed data in terms of the total variation.

To the best of our knowledge, we seem to first establish both the global convergence and global optimality of training GANs when the generator is parametrized by a neural network.

The file iclr2020_conference.pdf contains these instructions and illustrates the various formatting requirements your ICLR paper must satisfy.

Submissions must be made using L A T E X and the style files iclr2020_conference.sty and iclr2020_conference.bst (to be used with L A T E X2e).

The file iclr2020_conference.tex may be used as a "shell" for writing your paper.

All you have to do is replace the author, title, abstract, and text of the paper with your own.

The formatting instructions contained in these style files are summarized in sections 2, 3, and 4 below.

The text must be confined within a rectangle 5.5 inches (33 picas) wide and 9 inches (54 picas) long.

The left margin is 1.5 inch (9 picas).

Use 10 point type with a vertical spacing of 11 points.

Times New Roman is the preferred typeface throughout.

Paragraphs are separated by 1/2 line space, with no indentation.

Paper title is 17 point, in small caps and left-aligned.

All pages should start at 1 inch (6 picas) from the top of the page.

Authors' names are set in boldface, and each name is placed above its corresponding address.

The lead author's name is to be listed first, and the co-authors' names are set to follow.

Authors sharing the same address can be on the same line.

Please pay special attention to the instructions in section 4 regarding figures, tables, acknowledgments, and references.

The recommended paper length is 8 pages, with unlimited additional pages for citations.

There will be a strict upper limit of 10 pages for the main text.

Reviewers will be instructed to apply a higher standard to papers in excess of 8 pages.

Authors may use as many pages of appendices (after the bibliography) as they wish, but reviewers are not required to read these.

These instructions apply to everyone, regardless of the formatter being used.

Citations within the text should be based on the natbib package and include the authors' last names and year (with the "et al." construct for more than two authors).

When the authors or the publication are included in the sentence, the citation should not be in parenthesis using \citet{} (as in "See Hinton et al. (2006) for more information.").

Otherwise, the citation should be in parenthesis using \citep{} (as in "Deep learning shows promise to make progress towards AI (Bengio & LeCun, 2007) .").

The corresponding references are to be listed in alphabetical order of authors, in the REFERENCES section.

As to the format of the references themselves, any style is acceptable as long as it is used consistently.

Indicate footnotes with a number 1 in the text.

Place the footnotes at the bottom of the page on which they appear.

Precede the footnote with a horizontal rule of 2 inches (12 picas).

Make sure the figure caption does not get separated from the figure.

Leave sufficient space to avoid splitting the figure and figure caption.

You may use color figures.

However, it is best for the figure captions and the paper body to make sense if the paper is printed either in black/white or in color.

The Hessian matrix of f at input point x f (x)dx Definite integral over the entire domain of x S f (x)dx Definite integral with respect to x over the set S Probability and Information Theory P (a) A probability distribution over a discrete variable p(a) A probability distribution over a continuous variable, or over a variable whose type has not been specified

H (

Positive part of x, i.e., max(0, x) 1 condition is 1 if the condition is true, 0 otherwise

Do not change any aspects of the formatting parameters in the style files.

In particular, do not modify the width or length of the rectangle the text should fit into, and do not change font sizes (except perhaps in the REFERENCES section; see below).

Please note that pages should be numbered.

Please prepare PostScript or PDF files with paper size "US Letter", and not, for example, "A4".

The -t letter option on dvips will produce US Letter files.

Consider directly generating PDF files using pdflatex (especially if you are a MiKTeX user).

PDF figures must be substituted for EPS figures, however.

Otherwise, please generate your PostScript and PDF files with the following commands:

dvips mypaper.dvi -t letter -Ppdf -G0 -o mypaper.ps ps2pdf mypaper.ps mypaper.pdf

Most of the margin problems come from figures positioned by hand using \special or other commands.

We suggest using the command \includegraphics from the graphicx package.

Always specify the figure width as a multiple of the line width as in the example below using .eps graphics A number of width problems arise when LaTeX cannot properly hyphenate a line.

Please give LaTeX hyphenation hints using the \-command.

If you'd like to, you may include a section for author contributions as is done in many journals.

This is optional and at the discretion of the authors.

A APPENDIX You may include other additional sections here.

<|TLDR|>

@highlight

We establish global convergence to optimality for IPM-based GANs where the generator is an overparametrized neural network. 