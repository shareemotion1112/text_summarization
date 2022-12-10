The application of deep recurrent networks to audio transcription has led to impressive gains in automatic speech recognition (ASR) systems.

Many have demonstrated that small adversarial perturbations can fool deep neural networks into incorrectly predicting a specified target with high confidence.

Current work on fooling ASR systems have focused on white-box attacks, in which the model architecture and parameters are known.

In this paper, we adopt a black-box approach to adversarial generation, combining the approaches of both genetic algorithms and gradient estimation to solve the task.

We achieve a 89.25% targeted attack similarity after 3000 generations while maintaining 94.6% audio file similarity.

Figure 1 : Example of targeted adversarial attack on speech to text systems in practice combination of genetic algorithms and gradient estimation to solve this task.

The first phase of the 36 attack is carried out by genetic algorithms, which are a gradient-free method of optimization that 37 iterate over populations of candidates until a suitable sample is produced.

In order to limit excess 38 mutations and thus excess noise, we improve the standard genetic algorithm with a new momentum 39 mutation update.

The second phase of the attack utilizes gradient estimation, where the gradients 40 of individual audio points are estimated, thus allowing for more careful noise placement when the 41 adversarial example is nearing its target.

The combination of these two approaches provides a 89.25% 42 average targeted attack similarity with a 94.6% audio file similarity after 3000 generations.

43

Adversarial attacks can be created given a variety of information about the neural network, such as 45 the loss function or the output probabilities.

However in a natural setting, usually the neural network 46 behind such a voice control system will not be publicly released so an adversary will only have access 47 to an API which provides the text the system interprets given a continuous waveform.

Given this 48 constraint, we use the open sourced Mozilla DeepSpeech implementation as a black box system, 49 without using any information on how the transcription is done.

We perform our black box targeted attack on a model M given a benign input x and a target t by 51 perturbing x to form the adversarial input x = x + δ, such that M (x ) = t. To minimize the audible 52 noise added to the input, so a human cannot notice the target, we maximize the cross correlation 53 between x and x .

A sufficient value of δ is determined using our novel black box approach, so we 54 do not need access to the gradients of M to perform the attack.

Compared to images, audio presents a much more significant challenge for models to deal with.

While 57 convolutional networks can operate directly on the pixel values of images, ASR systems typically 58 require heavy pre-processing of the input audio.

Most commonly, the Mel-Frequency Cepstrum

(MFC) transform, essentially a fourier transform of the sampled audio file, is used to convert the 60 input audio into a spectogram which shows frequencies over time.

Models such as DeepSpeech (Fig. 61 2) use this spectogram as the initial input.

Extending the research done by BID0 , we propose a genetic algorithm and gradient estimation approach 81 to create targeted adversarial audio, but on the more complex DeepSpeech system.

The difficulty of 82 this task comes in attempting to apply black-box optimization to a deeply-layered, highly nonlinear 83 decoder model that has the ability to decode phrases of arbitrary length.

Nevertheless, the combination 84 of two differing approaches as well as the momentum mutation update bring new success to this task.

DeepSpeech outputs a probability distribution over all characters at every frame, for 50 frames per if EditDistance(t, Decode(best)) > 2 then // phase 1 -do genetic algorithm while populationSize children have not been made do Select parent1 from topk(population) according to sof tmax(their score) Select parent2 from topk(population) according to sof tmax(their score) child ← M utate(Crossover(parent1, parent2), p) end while newScores ← −CT CLoss(newPopulation, t) p ← M omentumU pdate(p, newScores, scores) else // phase 2 -do gradient estimation top-element ← top(population) grad-pop ← n copies of top-element, each mutated slightly at one index grad ← (−CT CLoss(grad-pop) − scores)/mutation-delta pop ← top-element + grad end if end while return best

As mentioned previously, Alzantot et al. BID0 demonstrated the success of a black-box adversarial 125 attack on speech-to-text systems using a standard genetic algorithm.

The basic premise of our 126 algorithm is that it takes in the benign audio sample and, through trial and error, adds noise to the 127 sample such that the perturbed adversarial audio is similar to the benign input yet is decoded as the Loss, we make modifications to the genetic algorithm and introduce our novel momentum mutation.

CTC-Loss, which as mentioned previously, is used to determine the similarity between an input audio 137 sequence and a given phrase.

We then form our elite population by selecting the best scoring samples 138 from our population.

The elite population contains samples with desirable traits that we want to carry 139 over into future generations.

We then select parents from the elite population and perform Crossover, 140 which creates a child by taking around half of the elements from parent1 and the other half from 141 parent2.

The probability that we select a sample as a parent is a function of the sample's score.

With some probability, we then add a mutation to our new child.

Finally, we update our mutation 143 probabilities according to our momentum update, and move to the next iteration.

The population will 144 continue to improve over time as only the best traits of the previous generations as well as the best 145 mutations will remain.

Eventually, either the algorithm will reach the max number of iterations, or 146 one of the samples is exactly decoded as the target, and the best sample is returned.

148 Algorithm 2 Mutation Input: Audio Sample x Mutation Probability p Output: Mutated Audio Sample x for all e in x do noise ← Sample(N (µ, σ 2 )) if Sample(Unif(0, 1))

< p then e ← e + f ilter highpass (noise) end if end for return xThe mutation step is arguably the most crucial component of the genetic algorithm and is our only 149 source of noise in the algorithm.

In the mutation step, with some probability, we randomly add noise 150 to our sample.

Random mutations are critical because it may cause a trait to appear that is beneficial 151 for the population, which can then be proliferated through crossover.

Without mutation, very similar 152 samples will start to appear across generations; thus, the way out of this local maximum is to nudge it 153 in a different direction in order to reach higher scores.

Furthermore, since this noise is perceived as background noise, we apply a filter to the noise before 155 adding it onto the audio sample.

The audio is sampled at a rate of f s = 16kHz, which means that 156 the maximum frequency response f max = 8kHz.

As seen by Reichenbach and Hudspeth [14] , given 157 that the human ear is more sensitive to lower frequencies than higher ones, we apply a highpass filter 158 at a cutoff frequency of f cutof f = 7kHz.

This limits the noise to only being in the high-frequency 159 range, which is less audible and thus less detectable by the human ear.

While mutation helps the algorithm overcome local maxima, the effect of mutation is limited by the 161 mutation probability.

Much like the step size in SGD, a low mutation probability may not provide 162 enough randomness to get past a local maximum.

If mutations are rare, they are very unlikely to 163 occur in sequence and add on to each other.

Therefore, while a mutation might be beneficial when 164 accumulated with other mutations, due to the low mutation probability, it is deemed as not beneficial 165 by the algorithm in the short term, and will disappear within a few iterations.

This parallels the step 166 size in SGD, because a small step size will eventually converge back at the local minimum/maximum.

However, too large of a mutation probability, or step size, will add an excess of variability and prevent 168 the algorithm from finding the global maximum/minimum.

To combat these issues, we propose

Momentum Mutation, which is inspired by the Momentum Update for Gradient Descent.

With this 170 update, our mutation probability changes in each iteration according to the following exponentially 171 weighted moving average update: DISPLAYFORM0 With this update equation, the probability of a mutation increases as our population fails to adapt 173 meaning the current score is close to the previous score.

The momentum update adds acceleration 174 to the mutation probability, allowing mutations to accumulate and add onto each other by keeping 175 the mutation probability high when the algorithm is stuck at a local maximum.

By using a moving 176 average, the mutation probability becomes a smooth function and is less susceptible to outliers in the 177 population.

While the momentum update may overshoot the target phrase by adding random noise, 178 overall it converges faster than a constant mutation probability by allowing for more acceleration in

Genetic algorithms work well when the target space is large and a relatively large number of mutation 182 directions are potentially beneficial; the strength of these algorithms lies in being able to search 183 large amounts of space efficiently BID7 .

When an adversarial sample nears its target perturbation, 184 this strength of genetic algorithms turn into a weakness, however.

Close to the end, adversarial 185 audio samples only need a few perturbations in a few key areas to get the correct decoding.

In this 186 case, gradient estimation techniques tend to be more effective.

Specifically, when edit distance of 187 the current decoding and the target decoding drops below some threshold, we switch to phase 2.

When approximating the gradient of a black box system, we can use the technique proposed by Nitin DISPLAYFORM0 Here, x refers to the vector of inputs representing the audio file.

δ i refers to a vector of all zeros,

Of the audio samples for which we ran our algorithm on, we achieved a 89.25% similarity between the 217 final decoded phrase and the target using Levenshtein distance, with an average of 94.6% correlation 218 similarity between the final adversarial sample and the original sample.

The average final Levenshtein 219 distance after 3000 iterations is 2.3, with 35% of the adversarial samples achieving an exact decoding 220 in less than 3000 generations, and 22% of the adversarial samples achieving an exact decoding in less 221 than 1000 generations.

One thing to note is that our algorithm was 35% successful in getting the decoded phrase to match 223 the target exactly; however, noting from figure 5, the vast majority of failure cases are only a few edit 224 distances away from the target.

This suggests that running the algorithm for a few more iterations 225 could produce a higher success rate, although at the cost of correlation similarity.

Indeed, it becomes 226 apparent that there is a tradeoff between success rate and audio similarity such that this threshold 227 could be altered for the attacker's needs.

One helpful visualization of the similarity between the original audio sample and the adversarial 233 audio sample through the overlapping of both waveforms, as shown in figure 4 .

As the visualization 234 shows, the audio is largely unchanged, and the majority of the changes to the audio is in the relatively 235 low volume noise applied uniformly around the audio sample.

This results in an audio sample that 236 still appears to transcribe to the original intended phrase when heard by humans, but is decoded as 237 the target adversarial phrase by the DeepSpeech model.

That 35% of random attacks were successful in this respect highlights the fact that black box 239 adversarial attacks are definitely possible and highly effective at the same time.

4 Conclusion

In combining genetic algorithms and gradient estimation we are able to achieve a black box adversarial 242 example for audio that produces better samples than each algorithm would produce individually.

By 243 initially using a genetic algorithm as a means of exploring more space through encouragement of 244 random mutations and ending with a more guided search with gradient estimation, we are not only 245 able to achieve perfect or near-perfect target transcriptions on most of the audio samples, we were able 246 to do so while retaining a high degree of similarity.

While this remains largely as a proof-of-concept 247 demonstration, this paper shows that targeted adversarial attacks are achievable on black box models 248 using straightforward methods.

Furthermore, the inclusion of momentum mutation and adding noise exclusively to high frequencies 250 improved the effectiveness of our approach.

Momentum mutation exaggerated the exploration at the 251 beginning of the algorithm and annealed it at the end, emphasizing the benefits intended by combining 252 genetic algorithms and gradient estimation.

Restricting noise to the high frequency domain improved 253 upon our similarity both subjectively by keeping it from interfering with human voice as well as 254 objectively in our audio sample correlations.

By combining all of these methods, we are able to 255 achieve our top results.

In conclusion, we introduce a new domain for black box attacks, specifically on deep, nonlinear 257 ASR systems that can output arbitrary length translations.

Using a combination of existing and novel 258 methods, we are able to exhibit the feasibility of our approach and open new doors for future research.

@highlight

We present a novel black-box targeted attack that is able to fool state of the art speech to text transcription.