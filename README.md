# avi
asynchronous variational integrators written in performant python, primarily numpy.  It makes use of the trimesh library to hold the current mesh in terms of tetrahedra.  The mechanics tend to be stable as the mechanical equilibrium is defined explicitely.  

visualisations are accessable in `doc/mov/`

functionality is in `nb/lib/`

To use machine learning to speed up computational mechanics, see "Future Directions" in the associated letter (see `doc/Adaptivly_Boosted_Time_Steps.pdf`).  This numerical method for adaptively boosted time steps was invented here for a class project that turned into a 9 month coding project.  I really enjoyed the class, which was a wonderful lecture series on geometric numerical integration given by Prof. Melvin Leok.

The original goal of this project was to use the finite element method to do general mechanical modeling of atrial function.  I would love to extend this project to patient-specific electromechanical modeling of quantitative cardiac pathophysiolgy in us humans.  With a relatively small amount of effort in python/numpy, I believe could be connected to things you might have done already.  If you are interested in this, I invite you to email me at tyree at physics dot ucsd dot edu.

After developing this finite element model, I think a different numerical method would be better suited for modeling atrial mechanics, as the human atrium is only ~1mm.  That being said, the human ventricals tend to be much thicker (~1cm), so I think the finite element model developed here could be more useful for modeling patient-specific left-ventricular ejection fraction.
