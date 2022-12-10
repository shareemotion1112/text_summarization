In this paper, we propose a novel approach to improve a given surface mapping through local refinement.

The approach receives an established mapping between two surfaces and follows four phases: (i) inspection of the mapping and creation of a sparse set of landmarks in mismatching regions; (ii) segmentation with a low-distortion region-growing process based on flattening the segmented parts; (iii) optimization of the deformation of segmented parts to align the landmarks in the planar parameterization domain; and (iv) aggregation of the mappings from segments to update the surface mapping.

In addition, we propose a new method to deform the mesh in order to meet constraints (in our case, the landmark alignment of phase (iii)).

We incrementally adjust the cotangent weights for the constraints and apply the deformation in a fashion that guarantees that the deformed mesh will be free of flipped faces and will have low conformal distortion.

Our new deformation approach, Iterative Least Squares Conformal Mapping (ILSCM), outperforms other low-distortion deformation methods.

The approach is general, and we tested it by improving the mappings from different existing surface mapping methods.

We also tested its effectiveness by editing the mappings for a variety of 3D objects.

C OMPUTING a cross-surface mapping between two surfaces (cross-parameterization) is a fundamental problem in digital geometric processing.

A wide range of methods have been developed to find such mappings [2] , [3] , [4] , but no single method results in a perfect mapping in every case.

Quite often, the mapping results may be good overall, but some specific, sometimes subtle, semantic features, such as articulations and facial features, may remain misaligned, as illustrated in Figure 1 .

These imperfections of the final result are often unacceptable in a production setting where the artist needs a high degree of control over the final result, and will often sacrifice automation of a method for higher control.

Typically, improving results using surface mapping methods requires the user to iteratively insert some landmarks and solve for the mapping globally.

However, since the imperfections are typically localized to a specific region, a local solution that does not change the mapping globally would be preferred in order to ensure that the method does not introduce artifacts elsewhere on the map.

This paper proposes a surface mapping editing approach providing local and precise control over the map adjustments.

The process begins with the inspection of an existing vertex-topoint surface mapping between two meshes.

In regions where the mapping exhibits some discrepancy, the user sets landmarks positioned at corresponding locations on both meshes.

For each such region, we extract a patch on both meshes in order to localize the changes in the mapping, and we flatten them on a common planar domain.

The mapping is improved based on a 2D deformation optimization that steers the landmarks toward correspondence while limiting distortion and having theoretical guarantees to maintain the local injectivity of the map.

We developed a new 2D deformation approach denoted Iterative Least Squares Conformal Maps (ILSCM), which iteratively minimizes a conformal energy, each iteration ensuring that flips do not occur, and in practice, ensuring progress toward satisfying the constraints.

We chose to work with a conformal energy as we want to be able to improve mappings where the deformation between the pair of meshes is not isometric.

Our editing approach can successfully align the mapping around landmarks without any degradation of the overall mapping.

The local surface maps are extracted from their respective deformed segments and parameterization domains, and are then combined to form an improved global surface mapping.

Our approach solves an important practical problem and offers three novel scientific contributions.

The first is a practical approach for local surface map editing, which we show, using both qualitative and quantitative metrics, provides better results than other stateof-the-art methods.

The second involves a compact segmentation which results in a compromise between a low-distortion flattening and a low-distortion deformation when aligning the landmarks.

The third is a new deformation approach, ILSCM, which preserves conformal energy better than other state-of-the-art methods, and that has theoretical guarantees preventing the introduction of foldovers.

While a lot of research has been done on creating correspondences between 3D objects, comparatively fewer methods have been proposed on correspondence editing.

Nguyen et al. [5] measure and optimize the consistency of sets of maps between pairs belonging to collections of surfaces.

They compute a score for the map, and then apply an optimization to iteratively improve the consistency.

The limitation of their method lies in its requirement of having multiple maps instead of a single map for a pair of surfaces.

Ovsjanikov et al. [6] propose the functional maps representation to establish correspondences between surfaces based on Laplacian eigenfunctions rather than points.

Working in that smooth basis function space makes it easy and efficient to generate smooth mappings, but significant modifications to the underlying method would be required to allow local adjustments of the mapping guided by the user.

Another limitation is that their method is limited to near-isometric surfaces since non-isometric deformation overrides the assumption that the change of basis matrix is sparse.

Ezuz and Ben-Chen [7] remove the isometric restriction in their proposed The initial mapping between mesh A and B is globally good, but is locally misaligned in the head region.

Using our approach, the mapping is locally improved (c).

method for the deblurring and denoising of functional maps.

They smooth a reconstructed map by mapping the eigenfunctions of the Laplacian of the target surface in the span of the source eigenfunctions.

Their technique can be incorporated into existing functional mapping methods, but selecting the right number of eigenfunctions to perform the denoising is difficult.

Compared with a ground truth mapping, increasing the number of eigenfunctions decreases the error until a minimum is reached, but adding more eigenfunctions beyond this point increases the error.

While this can be observed on a ground truth mapping, there are no methods to achieve the minimum error for an arbitrary mapping.

Gehre et al. [8] incorporate curve constraints into the functional map optimization to update the mapping between non-isometric surfaces.

They provide an interactive process by proposing a numerical method which optimizes the map with an immediate feedback.

While their method is not limited to the isometric surfaces, it does however need several curve constraints to obtain a meaningful functional map.

Vestner et al. [9] improve dense mappings even in the case of non-isometric deformations.

Their method is an iterative filtering scheme based on the use of geodesic Gaussian kernels.

An important restriction of their method is that it requires both surfaces to be discretized with the same number of vertices and vertex densities.

Panozzoet al. [10] propose the weighted averages (WA) on surfaces framework.

They use WA with landmarks in order to define mappings and then the user can improve the mapping by adjusting the landmarks.

Although WA generates good mapping results, its improved mapping application cannot use a mapping as input.

Since most state-of-the-art methods improve the mapping globally, this makes it hard for the user to fine-tune the mapping without risking modifying areas that should not be affected.

Furthermore, some methods face significant limitations such as being constrained to isometric deformations and requiring compatible meshes on both surfaces.

An alternative way to frame the correspondence editing problem is as a deformation method in a planar parameterization space.

Least Squares Conformal Maps (LSCM) [11] apply a deformation energy which contains a term for preserving surface features and a term for position constraints.

In contrast, Jacobson et al. [12] provide a smooth and shape-preserving deformation using biharmonic weights.

The main drawback of the LSCM and biharmonic weights methods is that they can introduce fold-overs while deforming the mesh.

Injectivity is a key property we want to achieve in our mapping editing, but extracting a mapping from meshes with fold-overs breaks this property.

The method of Chen and Weber [13] avoids fold-overs by applying inequality constraints solely to the boundaries.

Locally Injective Mappings (LIM) [14] and Scalable Locally Injective Mappings (SLIM) [15] propose a strategy where the energy of a triangle tends to infinity as the triangles become degenerate, and thus, any locally minimal solution will be by construction exempt of fold-overs.

While this is an elegant approach, the problem is that in some cases where, due to the user constraints, some triangles will come close to becoming degenerate, they carry a disproportionately high share of the total energy as compared to the rest of the triangles.

For our mapping editing application, such cases occur frequently, and we observed that in their presence, LIM and SLIM often produce an inferior result both qualitatively and quantitatively.

Gollaet al. [16] outperform LIM and SLIM by modifying the Newton iteration for the optimization of nonlinear energies on triangle meshes.

They analytically project the per-element Hessians to positive semidefinite matrices for efficient Newton iteration and apply global scaling to the initialization.

In this work, we propose to edit a surface mapping by locally adjusting the mapping in a bid to align landmarks set by the user.

To move the landmarks toward their expected positions, we deform a local segmented patch of the mesh.

We found that current deformation methods had drawbacks (flipped triangles and high distortion) forbidding their use in our mapping editing framework.

We thus derived a new deformation approach that iteratively minimizes a conformal energy, making sure that in each iteration we have no flipped triangles.

More specifically, our ILSCM approach optimizes the quadratic LSCM energy, but it relaxes the user constraints to avoid flips.

Therefore, after each iteration, the user constraints may not be satisfied, but by repeating the process, we reach a configuration that has low conformal energy (lower than LIM or SLIM), and the user constraints are guaranteed to be better satisfied than initially.

In practice, the user constraints are always satisfied up to a user-provided epsilon.

Our approach opens the door to a new family of deformation methods that optimize the deformation by effectively finding an optimal flow of the vertices based on conformal energy minimization.

As explained in the introduction, mappings computed by state-ofthe-art (automatic) methods are often globally good, but locally wrong in a few areas.

We provide an approach to locally improve the surface mapping.

The user typically inspects the mapping visually through texture transfer.

In local regions where the mapping should be improved, the user sets landmarks at locations that should correspond on the pair of surfaces (Fig. 2a) .

We edit the mapping by deforming parts of the meshes with respect to each other ( Fig. 2c ) to improve the alignment of the user-provided landmarks, and then we rebuild the mapping from the deformed parts (Fig. 2d) .

Our main goal is to obtain a low distortion of the meshes at each phase of our approach.

To deform the mapping with a good control on distortion, we conduct a planar parameterization of the meshes.

Since the planar parameterization of smaller segments of a mesh leads to less distortion versus when computed for the entire mesh, we do a segmentation based on user-identified local regions where the mapping needs to be updated.

Afterwards, we deform one segment with respect to the other in the planar parameterization space.

The deformation is aided by our new ILSCM, ensuring that the deformation causes limited distortion on the meshes.

Finally, the mapping is extracted based on how segments overlap each other.

Our approach has four key phases:

Our method works with input meshes A and B, along with a vertexto-point surface mapping between the meshes.

The mapping links each vertex of a mesh to a barycentric coordinate on the other mesh.

It should be noted that our approach works regardless of the initial method used to establish the surface mapping, as long as a dense mapping is provided.

Given a mapping, the user will visualize it using texture maps to identify mismatching regions.

These correspond to isolated zones of the meshes where the mapping is incorrect.

For each region, the user sets corresponding landmarks on both meshes at locations that should match each other.

The landmarks for each region i,

are expressed as barycentric coordinates on A and B, respectively.

The user provides hints where the mapping needs to be modified by setting pairs of landmarks on both meshes.

In order to keep the map editing local, a segment is identified on both meshes where the map editing will be performed.

Computing such a segment is not trivial as there are a number of requirements: the segment should be a single connected component with disk topology, should be compact, and should contain all the landmarks of the region i.

The size of the segment is also important.

If the segment is too large we may lose locality, but if it is too small, we may introduce further distortion if the vertices need to move over a long distance.

We assume that outside these segments, the mapping is satisfactory, and it can be used to set boundary conditions when deforming a segment with respect to the other to align the landmarks.

Our segmentation method has three steps.

In the first step (Fig. 3a) , we grow an initial patch on the 3D surface from the landmarks, ensuring that it is one connected component, that it encloses all of the landmarks, as well as the positions corresponding to the landmarks from the other mesh.

We flatten this patch in 2D (Fig. 3b) , where we have more tools available to control the size and shape of the patch.

In the second step, we compute a compact 2D patch from the convex hull of the landmarks in the 2D space (Fig. 3c) , and ensure that we fill any artificial internal boundaries.

In the third step, we grow the compact patch from the previous step to allow enough room between the boundary and the landmarks (Fig. 3d) , preparing the ground for a low-distortion deformation phase (Sec. 3.3).

This segmentation is applied to each region of meshes A and B independently.

We now explain in more detail the process as applied to one region i of mesh A, but the same is also conducted for mesh B and other regions.

Based on the mapping from mesh B to A, corresponding landmark positions are calculated on A for each landmark of L B (i),

The goal of the first step is to extract an initial patch and to flatten it.

To meet the two conditions of (1) having a single connected component and (2) containing all of the landmarks from L A (i) and CP B→A (i), we will compute the union of face groups computed by identifying faces around each landmark l A (i) j and around each corresponding position cp B→A (i) j .

For each, we iteratively add rings of faces until the group of faces contains at least half plus one of the landmarks from L A (i) and CP B→A (i).

The requirement to include half plus one of the landmarks ensures that when we combine the groups of faces, this initial patch meets the two conditions.

This procedure results in a "disk with holes" topology, which is sufficient to flatten the patch using ABF++ [17] .

One disadvantage of the initial patch is that it can contain concavities, and even internal "holes" with polygons from the full mesh missing.

From the initial patch of step one, the second step extracts a compact patch that surrounds the landmarks.

To this end, we identify the convex hull of the landmarks in the 2D parameterization space.

Then, we only consider the faces which have at least one of their vertices within the convex hull (faces identified in black in Fig. 3c ).

The use of the convex hull results in a patch exempt of large concavities in its boundary.

Nevertheless, depending on the meshes and the arrangement of landmarks, some of the initial patches have "holes" with polygons from the full mesh missing, creating artificial internal boundaries in the patch (Fig. 4) .

We add the missing faces by analyzing the inner boundaries (holes).

Filling the whole by adding the missing faces from the full mesh has the advantage of preventing unwanted deformation that would result from such artificial boundaries, and it ensures that there are no internal areas within the region where the mapping would not be adjusted.

The third step tries to balance the conflicting goals of having a small versus a large patch.

As can be seen in Fig. 5 , the larger the patch, the greater the distortion [18] between the patch in 3D and after flattening to 2D.

The distortion would be even higher if flattening the whole mesh (Fig. 6 ).

Conversely, a smaller patch means that the landmarks are closer to the boundary, and the deformation that aligns the landmarks will induce more distortion to the triangles between the boundary and the landmarks.

We thus want to grow the compact patch to ensure that there is enough room around each landmark and corresponding landmark position pairs to diffuse the distortion from the deformation phase.

We also know that as we get closer to the landmarks, we are getting closer to the areas where the mapping is wrong, and as such, extending outwards is necessary in order to have a good boundary condition for the deformation.

Regarding how far away the patch should be extended, we use a distance proportional to the geodesic distance (on the 3D mesh) between the landmark and its corresponding landmark position, adding faces that are within that distance from each landmark of the pair.

We compared the distortion between the 3D patch and the 2D patch for 1 to 10 times the geodesic distance.

Fig. 7 shows a pattern where very small patches do not have enough room to move the landmarks without high distortion.

At the same time, patches that are too large also exhibit large distortion because of the flattening from 3D to 2D.

A good compromise between the two sources of distortion is around two times the geodesic distance.

It is necessary to apply steps one and two, because step three alone could lead to disconnected components or artificial internal boundaries that do not exist in the mesh, but that exist in the patch because of the patch growth process (Fig. 4) .

The sequence of steps one to three provides the final segments from A, {A 1 , A 2 , . . .

, A n }, and the same for mesh B, yielding {B 1 , B 2 , . . .

, B n }.

These final segments are flattened through ABF++ and we refer to them as A 1 , A 2 , . . .

, A n and B 1 , B 2 , . . .

, B n .

We selected ABF++ to flatten our segments as we can assume it will yield a parameterization exempt of flipped triangles (injective).

Our main observation is that the initial surface mapping is globally adequate, but wrong in localized regions.

With this assumption, we line up the boundary of the regions by relying on the surface mapping.

We will then deform the interior to align the landmarks while keeping a low distortion.

As we have two segmented meshes, there are two ways to align the landmarks: deform A i on B i or B i on A i .

We select the one with the lower L 2 distortion [18] (between the segment in 3D and 2D), keeping it fixed and deforming the other.

Here, we will explain the deformation of B i with respect to A i , but the deformation of A i with respect to B i proceeds in the same way.

We deform B i in order to implicitly adjust the mapping by applying an energy minimization.

This is achieved by using positional user constraints (E L -aligning the landmarks, E B -aligning the boundary) coupled with a distortion preventing regularization (E D -globally deforming B i with low distortion), leading to the following equation:

The user constraints are by soft constraints as follows:

where k(i) are the landmarks of the segment i; l a j are the landmarks on A i ; vertices v ( j,1) , v ( j,2) , and v ( j,3) correspond to the three vertices of the triangle on B i containing the related landmark l b j ; and β ( j,1) , β ( j,2) , and β ( j,3) are the barycentric coordinates.

We use Ω( B i ) to denote the set of vertices on the boundary of B i , and map(v j ) to denote the corresponding position of v j on A i based on the mapping.

The energy E B pulls the vertices of the boundary of B i to the positions on A i where they correspond given the mapping.

When λ is small, the map will be injective, but the constraints are generally not satisfied (ultimately, if λ = 0, B i stays the same).

Conversely, when λ is large, the user constraints are satisfied, but flips may be introduced (λ = ∞ corresponds to using hard constraints).

An ideal deformation energy E D (V ) must meet three criteria: preserving the shape, maintaining the injectivity of the mapping (i.e., no flipped triangles), and satisfying the user constraints

.

This is an example where we want to adjust the mapping by moving the red landmarks to the positions of the corresponding green landmarks.

All deformations were done with the same patch boundary (the one from the ABF parameterization).

as much as possible.

For shape preservation, we experimented with several E D : LSCM [11] , LIM [14] , SLIM [15] , and KPNewton [16] .

Each of the energies has a number of pros and cons.

LSCM preserves the shape the best, but tends to introduce flips, as illustrated in Fig. 8 (c) .

LIM, SLIM, and KP-Newton on the other hand, guarantee injectivity (no flips), but introduce more distortion (between B i before and after deformation) than LSCM.

The graph in Fig. 9 illustrates these observations: LSCM has the least distortion, but flipped triangles would destroy the injectivity of the mapping.

LIM, SLIM, and KP-Newton have no flips, but overall, they have more distortion as compared to LSCM.

LIM minimizes a joint energy where one term optimizes the distortion and the second term optimizes the flips.

In such joint optimization frameworks, no one term may be close to a minimum of its own, as shown in Fig. 9 , where the results from LIM are worse in terms of the distortion energy than LSCM.

Fig. 10 .

This graph compares results in terms of the residual error of LSCM energy [11] for fixed values of λ .

The horizontal axis labels present λ and the number of iterations (λ , #iter).

The black bars highlight results with flipped triangles.

Since all these methods have shortcomings, we propose an approach that bridges the gap between the shape preservation of the original LSCM formulation with the injectivity preservation of LIM, SLIM, and KP-Newton.

Our approach, Iterative LSCM (ILSCM), is a different approach where we iteratively optimize to decrease E(V ), while preventing flips from occurring.

ILSCM performs iterative LSCM steps.

The first iteration uses the cotangent weights from segment B i .

The deformed segment from the first iteration is then used to set the weights for the second iteration, and so on.

At each iteration, if a triangle flip is detected, we decrease the value of λ and redo the same iteration.

This way, we are guaranteed to eventually find a λ that prevents flips from occurring.

We will now explain how we adaptively adjust λ to guarantee that we have no flips, while making as much progress as possible toward achieving the user constraints.

In order to measure if the constraints are satisfied, we consider the initial maximal distance between any landmark and corresponding landmark position pair dist 0 = max j l A (i) j − cp B→A (i) j , and iterate until the current maximal distance is below the threshold ε = dist 0 /250.

The Appendix A demonstrates that since the progression of landmarks is continuous with respect to λ , the approach will always find a λ that prevents having any flips and that enables progress toward the user constraints.

The progress could asymptotically stop, but in all cases, we are guaranteed to prevent triangles from flipping and we limit the mesh distortion.

For the mapping adjustment application, all of the examples we tested converged to meet the user constraints.

A larger λ will converge faster, but increases the likelihood of flipped triangles (Fig. 10, black bars) .

A small λ decreases the probability of flipped triangles, but increases the number of iterations needed to satisfy the user constraints.

As can be seen in Fig. 10 , whether using a small or large λ , the conformal residual is almost the same and it plateaus for smaller values of λ .

Consequently, even in the theoretical case where our approach would take very small incremental steps, the solution remains valid in the sense that it meets the constraints and the conformal residual remains close to the solution with larger steps.

In our experiments, we start with λ = 1000.

After each iteration, we automatically detect if there are flipped triangles, and if so, we redo the deformation of the iteration with λ = λ /2.

Fig. 11a demonstrates that the movement of a landmark is continuous with respect to different values of λ .

Fig. 11b further shows that even with small values of λ , we make progress toward satisfying the constraints.

We see that we make progress toward satisfying the user constraints even for small values of λ .

We also see that it is to our advantage to begin with a large value of λ to reduce the number of iterations before convergence.

As the last phase of our approach, we update the surface mapping between A and B from the planar parameterizations.

We first extract the mappings from each pair ( B i , A i ).

Then, we aggregate and transfer them to A and B. With the mapping being expressed as barycentric coordinates on the other mesh, we can update it by simply getting the barycentric coordinates of vertices from B i to faces of A i and vice-versa.

We validate our approach with various cases of faces, as well as with a wider range of objects with different morphologies from different data sets and artist contributions (see Table 1 ).

Experiments are presented based on different initial mapping methods: orbifold tutte embeddings [20] , Elastiface [2] , deformation transfer [21] , functional mapping method [3] , WA method [10] , and joint planar method [4] .

The number of landmarks and segments is proportional to the quality of the initial surface mapping and the complexity of the objects (see Table 1 ).

We evaluate the capabilities of our approach based on a qualitative evaluation by visual inspection and a quantitative evaluation based on geodesic distance.

This method prevents flipped triangles, and, essentially, it preserves the shape, while satisfying the user constraints.

Furthermore, it distributes the deformation error more uniformly across the mesh surface.

As can be seen in Fig. 9 , our distortion energy is lower than SLIM, is often lower than LIM, and is only slightly greater than LSCM.

We believe that this optimization strategy is more suited to this type of problem than a joint optimization strategy.

Fig. 12 compares LSCM, LIM, SLIM, and KP-Newton to our ILSCM (iterated 257 times and final λ was 31.25) for the example from Fig. 8 .

ILSCM distributed errors more uniformly over the whole deformed mesh, as compared to LIM, SLIM, and KP-Newton.

The accompanying video shows how our iterative approach progressively conducts the deformation, in comparison to LSCM, LIM, SLIM, and KP-Newton.

The meshes we deform in the video are the same as some of the examples from the LIM paper [14] .

For a fair comparison, we perform SLIM, LIM, and KP-Newton, all using the LSCM energy [11] .

For LIM, we apply a 1E−12 barrier weight, which is sufficient to prevent flips.

We experimented with barrier weights of LIM ranging from 1E−4 to 1E−20.

Barrier weights smaller than 1E−12 had an imperceptible impact, while those equal to or lower than 1E−20 did not converge.

For each deformation energy, we experimented with two different initial states: weights from the 3D triangles and weights from the flattened B i .

The distortion between B i before and after deformation was lowest when deforming using the weights from the flattened B i .

We thus used the weights from the flattened B i .

Visual inspection of results is a common form of validation for mapping problems.

We use a visualization method based on texture transfer.

We copy texture coordinates from one mesh to the other using mapping, setting the uv coordinates of a vertex to the barycentric interpolation of the uv coordinates of the other mesh.

For this visualization, we used two different types of textures.

The first type was a grid texture.

Figs. 1, 20 , and 13 qualitatively show that we obtain considerably better mappings using our editing approach.

An important assumption of our approach is that we can edit the mapping locally.

This implies that it is important to have a smooth transition at the boundary of the regions where we conduct local editing.

Fig. 14 shows a typical example of the smoothness of our edited mapping across the boundary of the segmented region.

The accompanying video also compares the transition of the mapping across the boundary by transferring texture from mesh A to mesh B using both initial mapping and edited mapping for the test cases of Fig. 13, Fig. 20 (top row), and Fig. 21 .

For the specific case of faces, we use realistic facial textures, making it easier to highlight important semantic facial features.

These features are derived from three important considerations: modeling, texturing, and animation.

A realistic facial texture is often enough to highlight modeling and texturing issues.

Problems around the nose (Figs. 15 and 23), lips (Figs. 15 and 23) , and These examples show cases that are ideal for our approach: the initial mappings are globally good, with few local misalignments.

Instead of solving for the mapping globally, our approach provides a local solution for these specific semantic regions.

For facial animation, other features need to be identified in the textures.

Accordingly, some of our texture visualizations use a set of curves that are positioned relative to the areas that deform during animation, based on the facial anatomy.

Fig. 15 illustrates the improvement in the correspondence of these animation-related features as compared against the initial surface mapping.

Our approach assumes that the segments can be flattened to 2D without any flipped triangles.

While the hypothesis is essential to get injective mappings, our approach is still robust to cases where the flattened segments would contain flipped faces.

Meshes used in the industry often exhibit small discrepancies such as cracks, holes, and handles.

Fig. 16 presents such a case where one of the meshes is of a different genera (contains two handles in the ear region).

Although it is not possible to get injective mappings when dealing with different genera, our approach behaves robustly: it can improve the mapping in the region with different genera and it does not degrade the mapping in the edited region nor in its vicinity.

Furthermore, even if it is not possible to achieve injective mappings in such cases, our edited mappings have reasonable properties: the mapping from the lower genera (A → B) is injective, and the mapping from the higher genera (B → A) is surjective.

While the qualitative evaluations of Sec. 4.1 demonstrate that our approach results in clear improvements, we also quantitatively measure how our approach improves the mappings.

We first use the same process as in the paper of Kim et al. [27] in order to measure the accuracy of the surface mapping.

Their method transfers vertex positions to the other mesh using the mapping under evaluation and a ground truth mapping.

It then computes the geodesic distances from the corresponding positions.

Fig. 17 shows the error of the initial mapping after our editing approach.

The comparative evaluation shown here relies on the ground truth mapping from SCAPE [24] (Fig. 17 (a) ) and TOSCA [25] (Fig. 17 (b) ) data sets.

We can see that applying our approach improves the mapping in the related regions without causing a degradation of the overall mapping.

Another way to measure the quality of a mapping is to morph a mesh into the shape of the other using the mapping.

Then, we evaluate the mapping by computing L 2 and L in f between the mesh and the morphed mesh to estimate the distortion which occurs in the mapping-based morphing process.

Fig. 18 shows the morphing of mesh A into mesh B using both the initial and new mappings.

With our updated mapping (Fig. 18 (d) ), the vertices of the head A are pulled back to the correct place.

This has the advantage of mapping the right density of vertices where needed, which is very important for morphing and in any transfer related to animation attributes (e.g., bones, vertex weights, and blend shapes).

Table 2 illustrates an evaluation of the quality of the edited mapping in comparison to the initial mapping between 3D shapes.

It shows that our edited mapping is as good as or better than the initial mapping when considering the distortion of the morphed mesh.

We can see that there is a single case where this measurement of distortion is slightly higher after the map is edited.

Even in this case, while the distortion is slightly higher, the edited mapping is clearly superior, as can be seen in Fig. 13.

We performed a qualitative comparison of the mapping editing versus the methods of Ezuz and Ben-Chen [7] .

We also did comparisons using LIM, SLIM, KM-Newton, and ILSCM to conduct the mapping editing.

We finally compared local editing to global editing using the method of Panozzo et al. [10] and the joint planar method [4] .

For the comparison to the method of Ezuz and Ben-Chen [7] , we established an initial mapping using a state-of-the-art functional mapping method [3] .

Note that we use the raw functional map, without the high-dimensional iterative closest point post-process refinement [6] .

Fig. 19 compares the mappings improved using our approach and the method of Ezuz and Ben-Chen [7] (which improves the mapping without any landmark).

Note how the added control of the landmarks provides a significantly improved mapping, exactly where intended.

Fig. 21 presents results when LIM, SLIM, KP-Newton, and ILSCM are used to conduct the mapping editing.

The comparison through texture transfer visualization shows that ILSCM is superior in adjusting the mapping as compared to LIM, SLIM, and KPNewton.

The accompanying video also compares LIM, SLIM, KP-Newton, and ILSCM in editing the mapping for the test case of Fig. 13 .

Adjusting the mapping globally requires having the initial constraints, the initial parameters of the method, and the method itself, which is constraining.

In addition, some mapping methods, such as that of Nogneng and Ovsjanikov [3] , do not let the user guide the process with landmarks, while others, such as OBTE [1] , only support a fixed number of landmarks (three or four landmarks for OBTE [1] ), which will be insufficient in many cases.

Furthermore, we believe that it is advantageous to ensure that changes occur locally, avoiding unexpected changes elsewhere in the mapping.

Fig. 20c (top row) shows that solving for the mapping globally is sometimes as effective as solving it locally.

Conversely, Fig. 20c (bottom row) shows that improving the mapping globally introduced artifacts on the fish head as compared to our local refinement (Fig. 20d bottom row) , which is exempt of such artifacts away from the edited region.

Fig. 22 compares the mappings improved using our approach as compared to solving globally using the WA method of Panozzo et al. [10] .

We established an initial mapping using the WA method.

Afterwards, with the WA method, we added two additional landmarks to improve the initial mapping.

For our method, we only consider the two new landmarks in improving the mapping.

It can be seen in Fig. 22 that editing the mapping locally was beneficial for this test case as well.

Several applications rely on a mapping between surfaces: texture transfer [26] , animation setup transfer [28] , and deformation transfer [21] .

We use the methods of Sumner et al. [21] and Avril et al. [28] to illustrate how the proposed approach can significantly improve the results of techniques relying on a mapping.

Fig. 23 shows a facial transfer result before and after editing.

Results demonstrate several issues and unpleasant deformations for fine features, such as strange deformations on the corners of the mouth.

With the corrected mapping, these problems disappear.

Fig. 24 shows a skeleton transfer [28] result before and after the mapping is edited.

Results demonstrate that the joint that was erroneously positioned outside of the thumb moves to the right place when improving the surface mapping locally in the thumb region instead of improving the mapping globally over the mesh.

Our approach works even for surfaces with boundaries inside the segments.

Such boundaries are commonly encountered with the ears, eyes, nostrils, and mouths of characters.

While we constrain the segment boundaries to prevent them from moving, an initial mesh boundary lying inside a segment will be free to move.

Leaving these inner boundaries completely free has a negative impact on the deformation.

Fig. 25 shows the deformation of the mouth without (c) and with (d) inner boundary fillings.

Note here the improvement of the mouth deformation when filling the inner boundary.

Our approach carves a new path in between the more classical shape-preserving methods, which often lose local injectivity, and the more current methods, which formulate the injectivity constraint as part of the optimization.

These latter approaches typically do not have a bound on the shape-preserving error.

In our approach, we are minimizing only the shape-preserving term (i.e., LSCM energy) and iteratively improving the user constraints while maintaining a locally injective map in each iteration.

We achieve this by carefully controlling the λ parameter in Eq. 1.

At one extreme, if λ is very large (i.e., infinity), the formulation is equivalent to the LSCM formulation.

If λ is very small, it takes many iterations for the user constraints to be satisfied, or in some cases, the user constraints may ultimately not be satisfied.

Our iterative scheme relies on two important observations.

If λ is 0, the solution is the same as the initial configuration.

Therefore, if we start in a locally injective configuration, the final result will be a locally injective configuration.

If the initial configuration is locally injective, there always exists a λ (however small) that will result in a locally injective configuration, where the user constraints are closer to the target.

This scheme will converge to a locally injective configuration.

Consequently, we iteratively repeat the optimization to fight against flipped faces, but convergence cannot be guaranteed.

It is always possible to design a landmark configuration in which the constraints cannot be met without flipped faces.

This is true for the other deformation methods as well.

Appendix B demonstrates different failure cases using different deformation methods.

In our experiments, the constraints are satisfied (up to numerical precision), even for extreme deformations.

In our results, we improved mappings which were initially computed from a variety of methods [1] , [3] , [4] , [10] , [21] , [26] .

Even if these initial mappings minimize different deformation energies, the fact that we rely on the LSCM conformal energy to edit them did not prevent our approach to improve the mappings.

One must keep in mind that the goal of the editing is not to strictly minimize a deformation energy, but to align important semantic features of the objects and maintain injectivity.

We analyzed our results to verify the degree to which the deformation deteriorates the shape of the triangles.

We checked 13 of the results found in this paper, and we considered that a detrimental deformation is one in which the angle becomes more than 20 times narrower after deformation.

Eleven cases had no such triangles, while the two other cases had two and three, respectively.

The worst triangle in our 13 test cases was 24 times narrower than before deformation.

Any deformation method is prone to result in thin triangles, so we compared our approach to LIM, SLIM, and KP-Newton for six examples.

When looking at the worst triangle found in the deformed meshes, ILSCM performed best for four of the test cases, while KP-Newton performed best for two of the test cases.

SLIM and LIM were systematically in third and fourth place behind ILSCM and KP-Newton.

Furthermore, our results were better than LIM, SLIM, and KP-Newton in terms of shape preservation and final triangulation, as can be seen in Fig. 12 and in the video.

We ran our experiments on a 3.40 GHz Intel Core-i7-4770 CPU with 12 GB of memory.

The presented approach was implemented with MATLAB, taking advantage of its sparse matrices and linear solvers.

Table 1 shows computation times for the segmentation and the deformation (including mapping extraction) phases.

Since our deformation phase is an iterative method, the time to edit a mapping depends on the size of the mismatching regions and iterations.

We have presented a novel approach for improving surface mappings locally.

Our approach is based on a low-distortion region-growing segmentation followed by an independent planar parameterization of each segment.

The mapping is then optimized based on an alignment of the user-prescribed landmarks in the parameterization space of each segment.

Our joint planar parameterization deformation for the segments is robust, and results in low distortion.

Our new iterative LSCM approach can be reused in several contexts where a deformation with low distortion is required.

From a practical perspective, our approach has several (a) Mesh A (b) Mesh B, initial mapping [10] (c) Mesh B, WA [10] (d) Mesh B, our edited mapping (e) Mesh A (f) Mesh B, initial mapping [10] (g) Mesh B, WA [10] (h) Mesh B, our edited mapping advantages.

It can be used to improve the mapping resulting from (a) Mesh A skeleton (b) Mesh B, initial skeleton [26] (c) Mesh B, edited skeleton Fig. 24 .

When using the mapping to retarget attributes, in this case the skeleton, an incorrect mapping will lead to problems, here putting the thumb joint outside of the mesh.

By locally editing the mapping, it is easy to fix such issues.

any surface mapping method.

It also provides a great deal of control, allowing the user to restrict editing to a specific region and to add as few or as many landmarks as necessary to achieve a desired result.

Our local editing leads to interesting questions which open many avenues for future work.

One such prospective area is higherlevel landmarks such as lines.

This will lead to challenges in terms of easing the interactive placement of these lines on both meshes, but will provide a better set of constraints for the deformation.

Another avenue would be to extend the scope to editing deformation transfer.

This will combine deformation with editing and enable the user to control animation retargeting.

To ensure that there is always a single solution, even if λ is arbitrarily small, we add a new term E B to Eq. 1:

where v old j denotes the position of vertex v j at the previous iteration.

The energy E B pulls the vertices of the boundary to where they correspond given the mapping.

The term E B pulls the vertices on the boundary of B i to their position at the previous iteration.

Eq. 3b is weighted by a small constant ξ = 0.001 such that in practice the vertices will converge to map(v j ).

The previous position v old j is initialized with the position on the boundary of the ABF of B i .

Our deformation method proceeds iteratively by finding a sequence of 2D embeddings V i of a given patch.

We show that if the initial embedding of the mesh V 0 has no fold-overs, then the resulting embedding at every iteration V i also has no fold-overs.

We prove this by induction.

The base case for i = 0 is given by the hypothesis, and thus, we are showing that if V i has no foldovers, our procedure will yield a configuration V i+1 that also has no fold-overs.

At every iteration, the new set of vertex positions V i+1 is obtained by solving Eq. 1: arg min

, where V i is the embedding in the current iteration, V i+1 is the new embedding we are computing, andλ > 0 is a parameter of the algorithm, constant w.r.t.

this minimization.

We selectλ as follows: we create a monotonically decreasing positive sequence λ j > 0 such that lim j→∞ (λ j ) = 0; we solve the optimization problem for the λ j in the sequence and stop at the first element in the sequenceλ = λ k that yields a fold-over free configuration, and we now show that such a λ k always exists.

Let B(x) ∈ R n×n , x ∈ R and B i j (x) is continuous in x ∀i, j. Lemma .1.

det(B(x)) is a continuous function in x.

Proof: We prove by induction on n. If n = 1, B(x) is a continuous real function.

det(B(x)) = B 1 1 (x) is also continuous.

We assume that the statement is true for n − 1 and we prove for n. We write det(B(x)) using the Laplace formula:

where M i j is the minor of the entry (i, j) defined as the determinant of the sub-matrix obtained by removing row i and column j from B. As each element of this matrix is also continuous in x and this reduced matrix is n − 1 × n − 1, it follows from the inductive hypothesis that M i j is also continuous in x. As det(B(x)) is obtained by using addition and multiplication of continuous functions, it follows that det(B(x)) is continuous in x.

Proof: If det(B(x)) = 0 ∀x then B(x) is invertible ∀x the inverse of a matrix has the following analytic expression:

where C is the matrix of co-factors: C i j = (−1) i+

j · M i j and M i j is the minor of the entry (i, j).

M i j (x) is continuous in x from Lemma .1.

It follows trivially that C i j is continuous in x.

Since det(B(x)) = 0 ∀x it follows from Corollary 1 that

is continuous in x. Since B −1 (x) is obtained by multiplying a scalar function continuous in x by a matrix whose entries are all continuous in x, it follows that B −1 i j (x) is continuous in x ∀i, j. V i+1 (λ ) is the minimizer of a quadratic energy function, and therefore, it has the standard least squares analytical solution:

where the matrix A and vector b are computed from Eq. 1 in a standard way for a least squares solution.

The matrix A(λ ) has the following structure:

where:

1) A 1 corresponds to E L and is a k × n matrix that encodes the landmark constraints of the patch.

These constraints are weighted by λ (Eq. 2a).

2) A 2 corresponds to E B and is a b × n matrix that encodes the boundary constraints of the patch.

These constraints are weighted by λ (Eq. 2b).

3) A 3 corresponds to E B , and is a 2×n matrix that constrains two boundary vertices to their positions in the previous iteration.

These constraints are weighted by a small constant ξ independent of λ (Eq. 2c).

4) A 4 corresponds to E D and is the m × n matrix from the original LSCM formulation [12] , where n is the number of vertices in the patch and m > n.

Lemma Proof:

The LSCM paper [12] shows that if we constrain exactly 2 vertices, we obtain a unique solution, which means that:

has rank n. Since the matrix A 3 4 does not depend on λ , it follows that rank(A 3 4 ) = n ∀λ .

Since the rank of A 3 4 cannot be larger than n, and the rank of the resulting matrix does not decrease by adding rows to the matrix, it follows that when stacking A 1 and A 2 to A 3 4 to form the final matrix A, rank(A(λ )) = n ∀λ .

Since the rank of the Gramm matrix is the same as that of the matrix, it follows that rank(A t (λ ) · A(λ )) = n ∀λ .

Since A t (λ ) · A(λ ) is a n × n matrix, this means that A has full rank, therefore det(A(λ )) = 0 ∀λ .

Lemma .5.

V i+1 (λ ) is continuous in λ .

Proof: λ = 0 reduces the linear system to only A 3 and A 4 .

Theorem .7.

∃k > 0 s.t.

V i+1 (λ k ) has no fold-overs.

Proof:

From Lemma .6, if λ = 0, then V i+1 has no fold-overs.

Since V i+1 is continuous in λ (Lemma .5), it follows that for all vertex positions ∃λ > 0 s.t.

∀λ , 0 < λ <λ ,V i+1 (λ ) has no foldovers.

Since the sequence λ j is monotonically decreasing and lim j→∞ (λ j ) = 0, it follows that ∃k s.t.0 < λ k <λ .

It follows that V i+1 (λ k ) has no fold-overs.

By proving Theorem .7, we show that at every iteration, our embedding V i+1 has no fold-overs and thus yields an injective map.

Our deformation method guarantees progress toward meeting the landmarks, constraints free of flipped faces, but it cannot guarantee that the user constraints will be satisfied.

In fact, there are cases where it is impossible to meet these constraints, such as the example in Fig. 26 .

There are also "hard" cases ( Fig. 27) where, while it might be possible to find a deformation that meets the constraints, deformation methods, such as LIM, SLIM, KP-Newton and our approach, are not able to find it.

@highlight

We propose a novel approach to improve a given cross-surface mapping through local refinement with a new iterative method to deform the mesh in order to meet user constraints.