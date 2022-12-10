Across numerous applications, forecasting relies on numerical solvers for partial differential equations (PDEs).

Although the use of deep-learning techniques has been proposed, the uses have been restricted by the fact the training data are obtained using PDE solvers.

Thereby, the uses were limited to domains, where the PDE solver was applicable, but no further.



We present methods for training on small domains, while applying the trained models on larger domains, with consistency constraints ensuring the solutions are physically meaningful even at the boundary of the small domains.

We demonstrate the results on an air-pollution forecasting model for Dublin, Ireland.

Solving partial differential equations (PDEs) underlies much of applied mathematics and engineering, ranging from computer graphics and financial pricing, to civil engineering and weather prediction.

Conventional approaches to prediction in PDE models rely on numerical solvers and require substantial computing resources in the model-application phase.

While in some application domains, such as structural engineering, the longer run-times may be acceptable, in domains with rapid decay of value of the prediction, such as weather forecasting, the run-time of the solver is of paramount importance.

In many such applications, the ability to generate large volumes of data facilitates the use of surrogate or reduced-order models BID3 obtained using deep artificial neural networks BID11 .

Although the observation that artificial neural networks could be applied to physical models is not new BID14 BID15 BID16 BID25 BID9 BID20 BID26 BID16 BID27 , and indeed, it is seen as one of the key trends BID2 BID13 BID31 on the interface of applied mathematics, data science, and deep learning, their applications did not reach the level of success observed in the field of the image classification, speech recognition, machine translation, and other problems processing unstructured high-dimensional data, yet.

A key issue faced by applications of deep-learning techniques to physical models is their scalability.

Even very recent research on deep-learning for physical models BID32 BID12 BID34 ) uses a solver for PDEs to obtain hundreds of thousands of outputs.

The deep learning can then be seen as means of non-linear regression between the inputs and outputs.

For example, BID12 have recently observed a factor of 12,000 computational speedup compared to that of a leading solver for the PDE, on the largest domain they were able to work with.

Considering the PDE solver is used to generate the outputs to train the deep-learning model on, however, the deep-learning model is limited to the domain and application that it is trained on.

We present methods for training Deep Neural Networks (DNNs) on small domains, while applying the trained models on larger domains, with consistency constraints ensuring the solutions are physically meaningful even at the boundaries of the small domains.

Our contributions are as follows:• definition of the consistency constraints, wherein the output for one (tile of a) mesh is used to constrain the output for another (tile of a) mesh.• methods for applying the consistency constraints within the training of a DNN, which allows for an increase in the extent of the spatial domain by concatenating the outputs of several PDE-based models by considering boundary conditions and state at the boundary.• a numerical study of the approach on a pollution-forecasting problem, wherein we lose accuracy from 1 to 7 per cent compared to the unconstrained model, but remove boundary artefacts.

We note that the methods can be applied both in terms of "patching" multiple (tiles of a) meshes, and in terms of "zooming" in multi-resolution approaches, where lower-resolution (e.g., city-, countryscale) component constrains higher-resolution components (e.g., district-, city-scale), which in turn impose consistency constraints on the former.

Our work is a first attempt to apply techniques based on domain decomposition to deep learning.

Conceptually, it promises the ability to concatenate outputs from disparate PDE-based simulation models into a single dataset for training deep learning with constraints to ensure consistency across the boundaries of the disparate simulations, i.e., physical viability across multiple meshes for a physical phenomenon governed by a PDE.

The approach under consideration is rather intuitive and consists of training a deep learning model for each available sub-grid, providing a method to ensure consistency across sub-grids, and scaling up to the wider area such that the accuracy of the predictions is increased.

Further, by enabling communication between meshes (via constraints), individual domain prediction can be provided with information external to the domain.

Let us consider an index-set M of meshes M m , m ∈ M, with sets of n m mesh points P (M m ).The output of each PDE-based simulation on such a mesh consists of values in R dm at each point DISPLAYFORM0 m of such points is of particular interest, which we call receptors R(M m ); the remainder of the points represents hidden points H(M m ).

The receptors and hidden points thus partition the mesh points DISPLAYFORM1 Further, let us consider the index-set B ⊆ M × M of boundaries B mn of meshes.

Such a possibly infinite boundary B mn ⊆ P (M m ) × P (M n ) links pairs of points from the two meshes.

To each boundary B mn we associate a constant mn that reflects the importance of this boundary.

Further, for each mesh M m we have an ordered set of simulations indexed with time t ∈ Z, where each simulation is defined by the inputs x t+k for some k > 0, in a recurrent fashion.

Our aim is to minimise residuals subject to consistency constraints to ensure physical "sanity" of the results, i.e., DISPLAYFORM2 ×|Q| is a projection operator that projects the array of outputs at all points onto the outputs at a subset of points Q ⊂ P (M m ), prox is a proximity operator, the decision variable defines the mapping DISPLAYFORM3 represents the output of a non-linear mapping between inputs and PDE-based simulation outputs at the points of the mesh, DISPLAYFORM4 ×nm , on each independent mesh M m , which can be seen as a nonlinear regression, and mn is a constant specific to (m, n) ∈ B. We provide examples of f (m) in the following sections.

The requirement of physical "sanity" is usually a statement about smoothness of the values of the mapping f (m) across the boundaries of two different meshes.

To be able to compare those values, we require that the dimensions are the same, that is ∀m, n ∈ M : DISPLAYFORM5 For example, for prox being the norm of a difference of the arguments, "smooth" at a point at the boundary of two meshes means that the values predicted within the two meshes at that point are numerically close to each other.

Also adding the norm of the difference of their gradients to that makes it a statement about the closeness of their first derivatives too.

Technically, "smoothness" is a statement about all their higher derivatives as well, however, we will only concern ourselves with their values, or zeroth order of derivatives, for now.

Notice though that generically this is an infinitely large problem.

In theory, equation 1 can be solved by Lagrangian relaxation techniques, e.g., by solving: DISPLAYFORM6 DISPLAYFORM7 .

This can be seen as a statement that there exist λ (m) t , t ∈ Z, such that the infimum over f (m) coincides with r * , for each m ∈ M. Clearly, if at least some of the boundaries B mn are infinite, then the optimisation problem is infinite-dimensional.

Further, one can borrow techniques from iterative solution schemes in the numerical analysis domain.

Notice that the first term in equation 2 is finite-dimensional and separable across the meshes.

For each mesh M m , m ∈ M, the above can be computed independently.

Further, one can subsample the boundaries to obtain a consistent estimator.

Subsequently, one could solve the finitedimensional projections of equation 2, wherein each new solution will increase the dimension of λ (m) t .

While this is feasible in theory, the inclusion of non-separable terms with λ (m) t would still render the solver less than practical.

Instead, we propose an iterative scheme, which is restricted to separable approximations.

Let us imagine that at time t, for a pair of points (p 1 , p 2 ) ∈ B mn on the boundary indexed with (m, n) ∈ B, we obtain values from the trained model at those points in the respective mesh, DISPLAYFORM8 .

While the two points p 1 , p 2 lay in two different meshes, we would like the outputs at those points to coincide.

For that we construct vectors χ p1,p2 and χ p1,p2 ∈ R d that serve as lower and upper bounds on the values obtained from the next iteration of the training of f (m) , that is, we element-wise construct DISPLAYFORM9 from which we can form univariate interval constraints, restricting the corresponding elements of both f (m) at p 1 and f (n) at p 2 of the next iteration to the respective interval χ p1,p2 i , χ p1,p2 i .

Notice that χ p1,p2 and χ p1,p2 become constant in the next iteration.

Further, notice also that replacing λ (m) t with a constant λ provides an upper bound on r * , which is computationally much easier to solve.

In the scheme, we consider a finite-dimensional projection of equation 2.

For each (m, n) ∈ B we consider a finite sampleB mn ⊂ B mn of pairs of points, for which we obtain DISPLAYFORM10 where we consider the function max : R × R d → R d to operate element-wise.

Further, when we consider λ as a hyperparameter, we obtain an optimisation problem separable in m ∈ M, which in the limit of |B mn | → |B mn | provides an over-approximation for any λ.

In deep learning, this scheme should be seen as a recurrent neural network (RNN).

A fundamental extension of RNN compared to traditional neural network approaches is parameter sharing across different parts of the model.

We refer to (Goodfellow et al., 2016, Chapter 10) introduction and FIG1 for a schematic illustration.

Each run provides constants (χ p1,p2 , χ p1,p2 ), which are used in the consistency constraints of the subsequent run.for an excellent f, λ χ χ s h 1 1 h 2 1 h 3 1 h q 1 h 1 2 h 2 2 h 3 2 h q 2 h 1 s h 2 s h 3 sIn terms of training the RNN, it is important to notice that equation 4 allows for very fast convergence rate even in many classes of non-linear maps f .

For instance, when DISPLAYFORM11 ×nm is a polynomial of a fixed degree BID10 , then equation 4 is strongly convex, despite the max function making it non-smooth.

The subgradient of the max function is well understood BID4 ) and readily implemented in major deep-learning frameworks.

Even in cases, when the resulting problem is not convex, one could consider convexifications, following BID19 .In numerical analysis, in general, and with respect to the multi-fidelity methods , in particular, our approach could be seen as iterative model-order reduction.

The original PDEs could be seen as the full-order model (FOM) to reduce, and equation 1 could be seen as a high-fidelity data-fit reduced-order model (ROM), albeit not a very practical one, whereas equation 4 could then be seen as a low-fidelity data-fit ROM, which allows for rapid prediction.

In learning theory, it is well known since the work of BID8 that even a feed-forward network with three or more layers of a sufficient number of neurons (e.g., with sigmoidal activation function) allows for a universal approximation of functions on a bounded interval.

It is not guaranteed, however, that the approximation has any further desirable properties, such as energy conservation etc.

Our consistency constraints allow for such properties.

Fundamentally, the approach can be summarised as learning the non-linear mapping between inputs and predictions on each independent mesh, and iterating to ensure consistency of the solution across meshes.

Such an approach draws on a long history of work on setting boundary conditions as consistency constraints in the solution of PDEs BID24 .

It can be applied to not only the simple patching of two tiles, but also when changing the resolution of the mesh.

We use the term patching for working with neighbouring meshes at a single resolution and zooming when the mesh resolution changes.

Both merging and zooming are illustrated in FIG3 .

To illustrate this framework, we train the Recurrent Neural Network for city-scale pollution monitoring, utilising:• The 3D structure of the atmosphere from our numerical weather forecasting model comprising the full atmospheric data (i.e., velocities, pressures, humidity, and temperatures in 3D).• Pollution measurements and traffic data, since traffic is measurable and strongly correlated to (esp.

nitrogen oxide, particulate matter) pollution in the cities.• The given discretisation of a city in multiple meshes, corresponding to multiple geographic areas with their specificities.

Our test case is based in the city of Dublin, Ireland, for which real-time streams of traffic and pollution data (from Dublin City Council), and weather data (from the Weather Company) are available to us, but which did not have any large-scale models of air pollution deployed.

Air pollution is known to have significant health impacts Organization (2018).

Typically, in cities, traffic-induced pollution is measured via the levels of nitrogen oxides (NOx) and Particulate Matter (PM).

The contribution of traffic to the levels of NOx is known to be around 70% in European cities, whereas the contribution of traffic to the levels of particulate matter pollution is known to be up to 50% in cities of OECD countries, in particular due to the heavy presence of diesel engines.

We aim at estimating and predicting the traffic-induced air pollution levels of NOx, PM2.5 and PM10, for defined receptors across the city.

An air pollution dispersion model propagates the pollution levels emitted from the roadway links (line sources).

The PDE-based model that we are using is based on the Gaussian Plume model, studied at least since the work of BID30 , and (by now) a standard model in describing the steady-state transport of pollutants.

The data inputs are the periodic traffic volumes for a number of roadway links across the city, and periodic updates of atmospheric data.

The outputs it provides are the estimates of pollution levels on a periodic basis.

For a comprehensive review of line source dispersion models, the interested reader may refer to BID21 BID29 .In addition to the traffic and weather data inputs, the Gaussian Plume model takes a lot of parameters as inputs, such as the emission factors associated to the roadway links (depending on the composition of the fleet), the pollution dispersion coefficients which are a proxy for modelling the terrain (density of buildings, etc.), and the background pollution levels (pollution that is not traffic induced).Such parameters are typically heterogeneous across cities and justify the use of different parameters, resolutions and physical resolution, hence PDE-based models, for the different meshes M m under consideration.

We use Caline 4 (at the California Department of Transportation), the open-source dispersion modelling suite, as a PDE solver to solve the Gaussian Plume model for the hourly inputs for each of the 12 domains described above.

We note while Caline is one of the "Preferred and Recommended Air Quality Dispersion Model" of the Environmental Protection Agency in the USA (US-EPA, 2018), it is limited to 20 line sources and 20 receptors per solve, which in turn forces an arbitrary inhomogeneous discretisation of the road network and is another motivation for the use of our deep learning approach.

We have implemented the approach for the use case of Dublin, Ireland.

There, the area is partitioned into 12 domains, with 10-20 line sources of pollution each mesh.

Time is discretised to hours.

For each hour, the inputs to the PDE solver comprise of traffic volume data at each line source, obtained from data aggregation of traffic loop detectors from the SCATS deployment in Dublin, and weather data at a discretisation of the spatial domain, obtained from The Weather Company under a licence: wind speed, wind direction, wind direction standard deviation, temperature, humidity.

Available training data comprises almost one year worth of hourly data from July 1st 2017 to April 31st 2018.

The outputs include concentrations of NO 2 (which is closely related to the NOx concentration), PM2.5 and PM10 concentrations at predefined receptors per domain, as suggested in FIG4 .

The parameters were chosen for each mesh M m based on the state-of-the-art practices: the emission factors based on the UK National Atmospheric Emissions Inventory database, dispersion coefficients based on the Caline recommendations (values for inner city, outer city areas), and background pollution levels chosen as the minimum time series values across the pollution measurements stations.

The RNN model is implemented in Tensorflow BID0 to obtain, in effect, the non-linear regression between the inputs and outputs, with the consistency constraints applied iteratively.

That is, with each map from the inputs to the outputs, we also obtain further consistency constraints to use by further runs on the same domain.

Crucially, we use domain knowledge to pick mn specific to (m, n) ∈ B based on the expected accuracy of the PDE-based model therein, as it is clear that a better accuracy can be expected when line sources are situated closer to the boundary.

We hence consider mn to be the maximum of 0.01 and the minimum distance of a line source to the boundary, where 0.01 corresponds to 100 meters.

This choice takes effect not only in the threshold in the inequality 1, but via the construction of , it also affects the "learning rate": the higher the mn , the faster the consistency constraints adapt to the solution obtained using the model trained on the adjacent mesh.

For validation purposes, we have use hourly NOx concentrations measured at 6 sites across the city.

(There are also 9 stations providing PM concentrations.)

FIG4 illustrates their positions, as well as the performance of the deep-learning forecaster at one example receptor, collocated with a measurement site used for our validation.

The mean absolute percentage error (MAPE) of the deep-learning forecast without the consistency constraints was about 1 per cent, which was reduced to 7 per cent using the consistency constraints.

This is because the same number of parameters have to fit an increased number of constraints, however, as can be seen from figure 5, the boundary artefacts disappeared after a few iterations of the training.

These values have been taken from a sample training for which we achieved convergence.

It will be interesting to optimise the algorithm architecture to observe convergence in more cases.

We have presented consistency constraints, which make it possible to train DNN on small domains and apply the trained models to larger domains while allowing incorporation of information external to the domain.

The consistency constraints will ensure the solutions are physically meaningful even at the boundary of the small domains in the output of the DNN.

We have demonstrated promising results on an air-pollution forecasting model for Dublin, Ireland.

The work is a first that makes possible numerous extensions.

First, one could consider further applications of the consistency constraints, e.g., in energy conservation, or in consider merging the outputs of a number of PDE models within multi-physics applications.

Second, in some applications, in may be useful to explore other network topologies.

Following BID34 , one could use long short-term memory (LSTM) units.

Further, over-fitting control could be based on an improved stacked auto-encoder architecture BID35 .

In interpretation of the trained model, the approach of BID7 may be applicable.

Our work could also be seen as an example of Geometric Deep Learning BID5 , especially in conjunction with the use of mesh-free methods BID28 , such as the 3D point clouds BID23 , non-uniform meshing, or non-uniform choice of receptors within the meshes.

Especially for applications, where the grids are in 3D or higher dimensions, the need for such techniques is clear.

More generally, one could explore links to isogeometric analysis of BID6 , which integrates solving PDEs with geometric modelling.

Finally, one could generalise our methods in a number of directions of the multi-fidelity modelling, e.g., by combining the reduced-order and full-order models using adaptation, fusion, or filtering.

Overall, the scaling up of deep learning for PDE-based models seems to be a particular fruitful area for further research.

Within the domain of our example application, recent surveys BID2 suggest that ours is the first use of deep learning in the forecasting of air pollution levels.

Following the copious literature on PDE-based models of air pollution, one could consider further pollutants such as ground-level ozone concentrations BID18 , and ensemble BID17 or multi-fidelity methods.

One may also consider a joint model, allowing for traffic forecasting, weather forecasting, and air pollution forecasting, within the same network, possibly using LSTM units BID7 , at the same time.

<|TLDR|>

@highlight

We present RNNs for training surrogate models of PDEs, wherein consistency constraints ensure the solutions are physically meaningful, even when the training uses much smaller domains than the trained model is applied to.