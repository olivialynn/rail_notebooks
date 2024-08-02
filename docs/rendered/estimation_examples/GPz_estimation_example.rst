GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.6048894e-01	 3.2561270e-01	-3.5076925e-01	 2.9898537e-01	[-3.0389794e-01]	 4.7707105e-01


.. parsed-literal::

       2	-2.8482741e-01	 3.1318997e-01	-2.5930126e-01	 2.8923447e-01	[-1.9106715e-01]	 2.3914051e-01


.. parsed-literal::

       3	-2.4009247e-01	 2.9244318e-01	-1.9676052e-01	 2.7505950e-01	[-1.2905969e-01]	 2.4965429e-01


.. parsed-literal::

       4	-1.9678880e-01	 2.7328005e-01	-1.4630530e-01	 2.6354593e-01	[-9.8302772e-02]	 2.3881364e-01
       5	-1.4723799e-01	 2.6028401e-01	-1.1762920e-01	 2.4874873e-01	[-5.3757983e-02]	 1.1832714e-01


.. parsed-literal::

       6	-8.3887061e-02	 2.5606546e-01	-5.6272818e-02	 2.4375061e-01	[-1.0424151e-02]	 1.1960077e-01
       7	-6.5149143e-02	 2.5195210e-01	-4.1523537e-02	 2.3916912e-01	[ 4.3610768e-03]	 1.1753249e-01


.. parsed-literal::

       8	-5.3647025e-02	 2.5004008e-01	-3.2917525e-02	 2.3725135e-01	[ 1.5600111e-02]	 1.1963177e-01
       9	-3.6495576e-02	 2.4677722e-01	-1.9164964e-02	 2.3478555e-01	[ 2.6721635e-02]	 1.1733556e-01


.. parsed-literal::

      10	-2.5655098e-02	 2.4463478e-01	-1.0289781e-02	 2.3384781e-01	[ 3.3833904e-02]	 1.2197423e-01
      11	-2.0025944e-02	 2.4365646e-01	-5.5567237e-03	 2.3331648e-01	[ 3.5739883e-02]	 1.1903191e-01


.. parsed-literal::

      12	-1.6234234e-02	 2.4290545e-01	-2.0777330e-03	 2.3307395e-01	[ 3.6364645e-02]	 1.2057638e-01
      13	-1.1577789e-02	 2.4196136e-01	 2.5152933e-03	 2.3260811e-01	[ 3.8892764e-02]	 1.1876321e-01


.. parsed-literal::

      14	 6.2819124e-02	 2.2891056e-01	 8.1868059e-02	 2.2198267e-01	[ 1.0492987e-01]	 2.3838520e-01


.. parsed-literal::

      15	 7.9360117e-02	 2.2633439e-01	 1.0059279e-01	 2.2030377e-01	[ 1.1284108e-01]	 2.3716545e-01
      16	 1.6656657e-01	 2.2025941e-01	 1.9131519e-01	 2.1736965e-01	[ 1.9924315e-01]	 1.1890912e-01


.. parsed-literal::

      17	 2.0689293e-01	 2.1645650e-01	 2.3876980e-01	 2.1289739e-01	[ 2.5826809e-01]	 1.2090850e-01
      18	 2.9354253e-01	 2.1343237e-01	 3.2527383e-01	 2.1155372e-01	[ 3.3459640e-01]	 1.1953878e-01


.. parsed-literal::

      19	 3.2186577e-01	 2.1057400e-01	 3.5370185e-01	 2.1103373e-01	[ 3.5922773e-01]	 1.2043357e-01
      20	 3.6368494e-01	 2.0736018e-01	 3.9712136e-01	 2.0826143e-01	[ 4.0346076e-01]	 1.1895776e-01


.. parsed-literal::

      21	 4.6913184e-01	 2.0641034e-01	 5.0237928e-01	 2.0828896e-01	[ 5.1010291e-01]	 1.1952615e-01
      22	 5.5733979e-01	 2.0521632e-01	 5.9396630e-01	 2.0719974e-01	[ 6.0031199e-01]	 1.1865449e-01


.. parsed-literal::

      23	 6.2753494e-01	 1.9546303e-01	 6.6671750e-01	 2.0081820e-01	[ 6.7398445e-01]	 1.2057543e-01
      24	 6.6539586e-01	 1.9191618e-01	 7.0392710e-01	 1.9550958e-01	[ 7.1461286e-01]	 1.1838245e-01


.. parsed-literal::

      25	 7.0005832e-01	 1.8978489e-01	 7.3825315e-01	 1.9090835e-01	[ 7.5338023e-01]	 1.2196398e-01
      26	 7.4736093e-01	 1.8476234e-01	 7.8601633e-01	 1.8468220e-01	[ 7.9563927e-01]	 1.2230301e-01


.. parsed-literal::

      27	 7.8878864e-01	 1.8926617e-01	 8.2795370e-01	 1.8656364e-01	[ 8.3267858e-01]	 1.2090349e-01
      28	 8.1918416e-01	 1.8953535e-01	 8.5874422e-01	 1.8614007e-01	[ 8.6375261e-01]	 1.1884785e-01


.. parsed-literal::

      29	 8.4775775e-01	 1.8998667e-01	 8.8815421e-01	 1.8759652e-01	[ 8.9193764e-01]	 1.2021685e-01
      30	 8.7967935e-01	 1.8593376e-01	 9.2095599e-01	 1.8463816e-01	[ 9.2818044e-01]	 1.1932683e-01


.. parsed-literal::

      31	 9.0591878e-01	 1.8089037e-01	 9.4699535e-01	 1.8312731e-01	[ 9.5172474e-01]	 1.2136269e-01
      32	 9.2705504e-01	 1.7531595e-01	 9.6898140e-01	 1.7605004e-01	[ 9.7167447e-01]	 1.1773682e-01


.. parsed-literal::

      33	 9.5204591e-01	 1.7134755e-01	 9.9422422e-01	 1.7279622e-01	[ 1.0012532e+00]	 1.2226772e-01
      34	 9.7697846e-01	 1.6693168e-01	 1.0199808e+00	 1.6774839e-01	[ 1.0216979e+00]	 1.2001204e-01


.. parsed-literal::

      35	 9.9989008e-01	 1.6430601e-01	 1.0443367e+00	 1.6471342e-01	[ 1.0419710e+00]	 1.2009764e-01
      36	 1.0150675e+00	 1.6291808e-01	 1.0605633e+00	 1.6183366e-01	[ 1.0573411e+00]	 1.1891127e-01


.. parsed-literal::

      37	 1.0287946e+00	 1.6141212e-01	 1.0744258e+00	 1.6106212e-01	[ 1.0678672e+00]	 1.2123394e-01
      38	 1.0430788e+00	 1.5936291e-01	 1.0888148e+00	 1.5983352e-01	[ 1.0823212e+00]	 1.1934161e-01


.. parsed-literal::

      39	 1.0626851e+00	 1.5662795e-01	 1.1087978e+00	 1.5713254e-01	[ 1.0981796e+00]	 1.2133408e-01
      40	 1.0730758e+00	 1.5608102e-01	 1.1198223e+00	 1.5771970e-01	[ 1.1063376e+00]	 1.1827421e-01


.. parsed-literal::

      41	 1.0855394e+00	 1.5537476e-01	 1.1316998e+00	 1.5687514e-01	[ 1.1142927e+00]	 1.2063980e-01
      42	 1.0950072e+00	 1.5435788e-01	 1.1412940e+00	 1.5475041e-01	[ 1.1209123e+00]	 1.2024736e-01


.. parsed-literal::

      43	 1.1102623e+00	 1.5265958e-01	 1.1568798e+00	 1.5105722e-01	[ 1.1293176e+00]	 1.1973810e-01
      44	 1.1272274e+00	 1.5045320e-01	 1.1739036e+00	 1.4629658e-01	[ 1.1471646e+00]	 1.1838913e-01


.. parsed-literal::

      45	 1.1412136e+00	 1.4921375e-01	 1.1879843e+00	 1.4400976e-01	[ 1.1572029e+00]	 1.2073469e-01
      46	 1.1521913e+00	 1.4837724e-01	 1.1990757e+00	 1.4286486e-01	[ 1.1697810e+00]	 1.2214994e-01


.. parsed-literal::

      47	 1.1653080e+00	 1.4756653e-01	 1.2128638e+00	 1.4108541e-01	[ 1.1824730e+00]	 1.1981082e-01
      48	 1.1737932e+00	 1.4649608e-01	 1.2219401e+00	 1.3993859e-01	[ 1.1963409e+00]	 1.1850810e-01


.. parsed-literal::

      49	 1.1819918e+00	 1.4606079e-01	 1.2300513e+00	 1.3973976e-01	[ 1.2009548e+00]	 1.2046361e-01
      50	 1.1929125e+00	 1.4504652e-01	 1.2412032e+00	 1.3892578e-01	[ 1.2042141e+00]	 1.2721157e-01


.. parsed-literal::

      51	 1.2028338e+00	 1.4409659e-01	 1.2512527e+00	 1.3840019e-01	[ 1.2105808e+00]	 1.2063670e-01
      52	 1.2212405e+00	 1.4245867e-01	 1.2703419e+00	 1.3711751e-01	[ 1.2205131e+00]	 1.1876512e-01


.. parsed-literal::

      53	 1.2239576e+00	 1.4132807e-01	 1.2735914e+00	 1.3711036e-01	[ 1.2248319e+00]	 1.1967754e-01
      54	 1.2414678e+00	 1.4062954e-01	 1.2902856e+00	 1.3637636e-01	[ 1.2401430e+00]	 1.3072562e-01


.. parsed-literal::

      55	 1.2481526e+00	 1.4030292e-01	 1.2971008e+00	 1.3637423e-01	[ 1.2423406e+00]	 1.2012792e-01
      56	 1.2583006e+00	 1.3949164e-01	 1.3075384e+00	 1.3638098e-01	[ 1.2470184e+00]	 1.1859775e-01


.. parsed-literal::

      57	 1.2684010e+00	 1.3847152e-01	 1.3179467e+00	 1.3586254e-01	[ 1.2523052e+00]	 1.1940360e-01
      58	 1.2799565e+00	 1.3678418e-01	 1.3299178e+00	 1.3473182e-01	[ 1.2626409e+00]	 1.2625289e-01


.. parsed-literal::

      59	 1.2887853e+00	 1.3633337e-01	 1.3386382e+00	 1.3406273e-01	[ 1.2711033e+00]	 1.2027645e-01
      60	 1.2961965e+00	 1.3602840e-01	 1.3460728e+00	 1.3352864e-01	[ 1.2790642e+00]	 1.1864662e-01


.. parsed-literal::

      61	 1.3032255e+00	 1.3567621e-01	 1.3532902e+00	 1.3351982e-01	[ 1.2840883e+00]	 1.2019420e-01
      62	 1.3111411e+00	 1.3496192e-01	 1.3612047e+00	 1.3286232e-01	[ 1.2961432e+00]	 1.1932731e-01


.. parsed-literal::

      63	 1.3197162e+00	 1.3437346e-01	 1.3699261e+00	 1.3277677e-01	[ 1.3016827e+00]	 1.2044191e-01
      64	 1.3295976e+00	 1.3363832e-01	 1.3800572e+00	 1.3242493e-01	[ 1.3117255e+00]	 1.1840582e-01


.. parsed-literal::

      65	 1.3355671e+00	 1.3334468e-01	 1.3862978e+00	 1.3232058e-01	  1.3080569e+00 	 1.1965513e-01
      66	 1.3429701e+00	 1.3323683e-01	 1.3935215e+00	 1.3209205e-01	[ 1.3163286e+00]	 1.1795211e-01


.. parsed-literal::

      67	 1.3494607e+00	 1.3288208e-01	 1.4000290e+00	 1.3178433e-01	[ 1.3207807e+00]	 1.2870479e-01
      68	 1.3556947e+00	 1.3276810e-01	 1.4064411e+00	 1.3161521e-01	[ 1.3209832e+00]	 1.1760736e-01


.. parsed-literal::

      69	 1.3618912e+00	 1.3181383e-01	 1.4130677e+00	 1.3104682e-01	  1.3209352e+00 	 1.1958647e-01
      70	 1.3704524e+00	 1.3145787e-01	 1.4215057e+00	 1.3082397e-01	[ 1.3287891e+00]	 1.1821532e-01


.. parsed-literal::

      71	 1.3764069e+00	 1.3079763e-01	 1.4275857e+00	 1.3059611e-01	[ 1.3324648e+00]	 1.2026215e-01
      72	 1.3811139e+00	 1.3011373e-01	 1.4326934e+00	 1.3041759e-01	  1.3304593e+00 	 1.1852837e-01


.. parsed-literal::

      73	 1.3862941e+00	 1.2944557e-01	 1.4379156e+00	 1.3006722e-01	[ 1.3342305e+00]	 1.1952901e-01
      74	 1.3920976e+00	 1.2888434e-01	 1.4437839e+00	 1.2975904e-01	[ 1.3358650e+00]	 1.1849236e-01


.. parsed-literal::

      75	 1.3991876e+00	 1.2839927e-01	 1.4508992e+00	 1.2943409e-01	  1.3352189e+00 	 1.2684488e-01
      76	 1.4050114e+00	 1.2767919e-01	 1.4568104e+00	 1.2901885e-01	  1.3290849e+00 	 1.2141061e-01


.. parsed-literal::

      77	 1.4107246e+00	 1.2762600e-01	 1.4623115e+00	 1.2876532e-01	  1.3358384e+00 	 1.1985779e-01
      78	 1.4143463e+00	 1.2747241e-01	 1.4659224e+00	 1.2853194e-01	[ 1.3367604e+00]	 1.1839676e-01


.. parsed-literal::

      79	 1.4181014e+00	 1.2728314e-01	 1.4697819e+00	 1.2814785e-01	[ 1.3397064e+00]	 1.2067127e-01
      80	 1.4220066e+00	 1.2707263e-01	 1.4737539e+00	 1.2787453e-01	[ 1.3398272e+00]	 1.1808443e-01


.. parsed-literal::

      81	 1.4263092e+00	 1.2682654e-01	 1.4781316e+00	 1.2771774e-01	  1.3387110e+00 	 1.1978817e-01
      82	 1.4324109e+00	 1.2650526e-01	 1.4844047e+00	 1.2717387e-01	[ 1.3419154e+00]	 1.1799598e-01


.. parsed-literal::

      83	 1.4361519e+00	 1.2639750e-01	 1.4881117e+00	 1.2727131e-01	  1.3391356e+00 	 1.2001705e-01
      84	 1.4387039e+00	 1.2624813e-01	 1.4905923e+00	 1.2717462e-01	  1.3411573e+00 	 1.2511468e-01


.. parsed-literal::

      85	 1.4426753e+00	 1.2608523e-01	 1.4946403e+00	 1.2690499e-01	  1.3417227e+00 	 1.1988068e-01
      86	 1.4464915e+00	 1.2591334e-01	 1.4986471e+00	 1.2660420e-01	  1.3377785e+00 	 1.1811304e-01


.. parsed-literal::

      87	 1.4506631e+00	 1.2569581e-01	 1.5031053e+00	 1.2613277e-01	  1.3392417e+00 	 1.2064981e-01
      88	 1.4546772e+00	 1.2551894e-01	 1.5071607e+00	 1.2573627e-01	  1.3409575e+00 	 1.1845374e-01


.. parsed-literal::

      89	 1.4578313e+00	 1.2526380e-01	 1.5102598e+00	 1.2554810e-01	[ 1.3440601e+00]	 1.2062025e-01
      90	 1.4610251e+00	 1.2503278e-01	 1.5134222e+00	 1.2518918e-01	[ 1.3493244e+00]	 1.1840463e-01


.. parsed-literal::

      91	 1.4659127e+00	 1.2454669e-01	 1.5182911e+00	 1.2447206e-01	[ 1.3539343e+00]	 1.2034798e-01
      92	 1.4688220e+00	 1.2445388e-01	 1.5212301e+00	 1.2420343e-01	  1.3532777e+00 	 1.2718368e-01


.. parsed-literal::

      93	 1.4710969e+00	 1.2440495e-01	 1.5234224e+00	 1.2404984e-01	[ 1.3581827e+00]	 1.2064028e-01
      94	 1.4732023e+00	 1.2437243e-01	 1.5255099e+00	 1.2397562e-01	[ 1.3599125e+00]	 1.1839533e-01


.. parsed-literal::

      95	 1.4762942e+00	 1.2425242e-01	 1.5286564e+00	 1.2388180e-01	[ 1.3619069e+00]	 1.2080169e-01
      96	 1.4782775e+00	 1.2413780e-01	 1.5308846e+00	 1.2376648e-01	  1.3555094e+00 	 1.1844635e-01


.. parsed-literal::

      97	 1.4825857e+00	 1.2401780e-01	 1.5351493e+00	 1.2364117e-01	[ 1.3619947e+00]	 1.1942720e-01
      98	 1.4842675e+00	 1.2390351e-01	 1.5368134e+00	 1.2352378e-01	[ 1.3637660e+00]	 1.1983371e-01


.. parsed-literal::

      99	 1.4867508e+00	 1.2375984e-01	 1.5393624e+00	 1.2326711e-01	[ 1.3645546e+00]	 1.2119818e-01
     100	 1.4888801e+00	 1.2356202e-01	 1.5416269e+00	 1.2279734e-01	[ 1.3651040e+00]	 1.2864685e-01


.. parsed-literal::

     101	 1.4915389e+00	 1.2351267e-01	 1.5442853e+00	 1.2260509e-01	[ 1.3670982e+00]	 1.2051916e-01
     102	 1.4935335e+00	 1.2347170e-01	 1.5463055e+00	 1.2244211e-01	[ 1.3681798e+00]	 1.1836505e-01


.. parsed-literal::

     103	 1.4954941e+00	 1.2340326e-01	 1.5483118e+00	 1.2226813e-01	[ 1.3688365e+00]	 1.2013745e-01
     104	 1.4983117e+00	 1.2309371e-01	 1.5513143e+00	 1.2199021e-01	  1.3661196e+00 	 1.1921644e-01


.. parsed-literal::

     105	 1.5011742e+00	 1.2299509e-01	 1.5542240e+00	 1.2166873e-01	  1.3664590e+00 	 1.2051725e-01
     106	 1.5027414e+00	 1.2284149e-01	 1.5557168e+00	 1.2164847e-01	  1.3683518e+00 	 1.1858582e-01


.. parsed-literal::

     107	 1.5052907e+00	 1.2255940e-01	 1.5582135e+00	 1.2156376e-01	  1.3647747e+00 	 1.2053013e-01
     108	 1.5073092e+00	 1.2222132e-01	 1.5602120e+00	 1.2136170e-01	  1.3642784e+00 	 1.1823535e-01


.. parsed-literal::

     109	 1.5092719e+00	 1.2214610e-01	 1.5621454e+00	 1.2132935e-01	  1.3601300e+00 	 1.2954974e-01
     110	 1.5113558e+00	 1.2207146e-01	 1.5642377e+00	 1.2123670e-01	  1.3553977e+00 	 1.1899877e-01


.. parsed-literal::

     111	 1.5131950e+00	 1.2191011e-01	 1.5660831e+00	 1.2118527e-01	  1.3542204e+00 	 1.2006164e-01
     112	 1.5146298e+00	 1.2140862e-01	 1.5676966e+00	 1.2111569e-01	  1.3512473e+00 	 1.1926436e-01


.. parsed-literal::

     113	 1.5175605e+00	 1.2131689e-01	 1.5705523e+00	 1.2110422e-01	  1.3523499e+00 	 1.1952090e-01
     114	 1.5184914e+00	 1.2124668e-01	 1.5714613e+00	 1.2109958e-01	  1.3543652e+00 	 1.1794949e-01


.. parsed-literal::

     115	 1.5207145e+00	 1.2099958e-01	 1.5737321e+00	 1.2103150e-01	  1.3546210e+00 	 1.2014699e-01
     116	 1.5222572e+00	 1.2072817e-01	 1.5754013e+00	 1.2105145e-01	  1.3528715e+00 	 1.1831927e-01


.. parsed-literal::

     117	 1.5240738e+00	 1.2067626e-01	 1.5771854e+00	 1.2102628e-01	  1.3508864e+00 	 1.2993765e-01
     118	 1.5252148e+00	 1.2067933e-01	 1.5783051e+00	 1.2097905e-01	  1.3495918e+00 	 1.1906505e-01


.. parsed-literal::

     119	 1.5267546e+00	 1.2063296e-01	 1.5798541e+00	 1.2099053e-01	  1.3472682e+00 	 1.1979318e-01
     120	 1.5288279e+00	 1.2057637e-01	 1.5819671e+00	 1.2093816e-01	  1.3460219e+00 	 1.1846018e-01


.. parsed-literal::

     121	 1.5307919e+00	 1.2041299e-01	 1.5839765e+00	 1.2108683e-01	  1.3420686e+00 	 1.1953163e-01
     122	 1.5322563e+00	 1.2028982e-01	 1.5854802e+00	 1.2113867e-01	  1.3425817e+00 	 1.1821699e-01


.. parsed-literal::

     123	 1.5338269e+00	 1.2019118e-01	 1.5871055e+00	 1.2116447e-01	  1.3377109e+00 	 1.2008452e-01
     124	 1.5352701e+00	 1.2017704e-01	 1.5885629e+00	 1.2103054e-01	  1.3410554e+00 	 1.1928892e-01


.. parsed-literal::

     125	 1.5366314e+00	 1.2022047e-01	 1.5899019e+00	 1.2093561e-01	  1.3372371e+00 	 1.2807322e-01
     126	 1.5382456e+00	 1.2036378e-01	 1.5914694e+00	 1.2072411e-01	  1.3315566e+00 	 1.2011790e-01


.. parsed-literal::

     127	 1.5389813e+00	 1.2043988e-01	 1.5921888e+00	 1.2065841e-01	  1.3271974e+00 	 1.1968780e-01
     128	 1.5399824e+00	 1.2043055e-01	 1.5931623e+00	 1.2067719e-01	  1.3278565e+00 	 1.1872864e-01


.. parsed-literal::

     129	 1.5415409e+00	 1.2040863e-01	 1.5947118e+00	 1.2075269e-01	  1.3277358e+00 	 1.1995459e-01
     130	 1.5424856e+00	 1.2037001e-01	 1.5956665e+00	 1.2080498e-01	  1.3284931e+00 	 1.1837149e-01


.. parsed-literal::

     131	 1.5446570e+00	 1.2039400e-01	 1.5979021e+00	 1.2088096e-01	  1.3302350e+00 	 1.2091136e-01
     132	 1.5457907e+00	 1.2026112e-01	 1.5991447e+00	 1.2114912e-01	  1.3271361e+00 	 1.2048960e-01


.. parsed-literal::

     133	 1.5472368e+00	 1.2028036e-01	 1.6005261e+00	 1.2102015e-01	  1.3309355e+00 	 1.2063074e-01
     134	 1.5481484e+00	 1.2026532e-01	 1.6014397e+00	 1.2096388e-01	  1.3310001e+00 	 1.2838840e-01


.. parsed-literal::

     135	 1.5494144e+00	 1.2018749e-01	 1.6027428e+00	 1.2099174e-01	  1.3272791e+00 	 1.2095928e-01
     136	 1.5515975e+00	 1.1999544e-01	 1.6049971e+00	 1.2121718e-01	  1.3161963e+00 	 1.1917210e-01


.. parsed-literal::

     137	 1.5524287e+00	 1.1984948e-01	 1.6058798e+00	 1.2138561e-01	  1.3050282e+00 	 2.3819089e-01
     138	 1.5539817e+00	 1.1969479e-01	 1.6074613e+00	 1.2165621e-01	  1.2964447e+00 	 1.1831570e-01


.. parsed-literal::

     139	 1.5554796e+00	 1.1955749e-01	 1.6089703e+00	 1.2185530e-01	  1.2908854e+00 	 1.1976695e-01
     140	 1.5572189e+00	 1.1938336e-01	 1.6107460e+00	 1.2205577e-01	  1.2860489e+00 	 1.1882401e-01


.. parsed-literal::

     141	 1.5579247e+00	 1.1925860e-01	 1.6115146e+00	 1.2216136e-01	  1.2828919e+00 	 1.2856150e-01
     142	 1.5590959e+00	 1.1927633e-01	 1.6126454e+00	 1.2204808e-01	  1.2848035e+00 	 1.1921811e-01


.. parsed-literal::

     143	 1.5596627e+00	 1.1926836e-01	 1.6132193e+00	 1.2198528e-01	  1.2849027e+00 	 1.2051439e-01
     144	 1.5604357e+00	 1.1923569e-01	 1.6140203e+00	 1.2196410e-01	  1.2821859e+00 	 1.1875701e-01


.. parsed-literal::

     145	 1.5618523e+00	 1.1917543e-01	 1.6154688e+00	 1.2205346e-01	  1.2746251e+00 	 1.1953926e-01
     146	 1.5625927e+00	 1.1906291e-01	 1.6163348e+00	 1.2233820e-01	  1.2503706e+00 	 1.1787462e-01


.. parsed-literal::

     147	 1.5646660e+00	 1.1908103e-01	 1.6183251e+00	 1.2245975e-01	  1.2521157e+00 	 1.2032104e-01
     148	 1.5655299e+00	 1.1908398e-01	 1.6191392e+00	 1.2253666e-01	  1.2522213e+00 	 1.1798835e-01


.. parsed-literal::

     149	 1.5671364e+00	 1.1909623e-01	 1.6207069e+00	 1.2274752e-01	  1.2457114e+00 	 1.2943959e-01
     150	 1.5690843e+00	 1.1910381e-01	 1.6226372e+00	 1.2294323e-01	  1.2339948e+00 	 1.1878586e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 35.4 s, sys: 41.7 s, total: 1min 17s
    Wall time: 19.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f9d388f4ac0>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 10000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10000 - 20000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20000 - 20449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 814 ms, sys: 990 ms, total: 1.8 s
    Wall time: 584 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: ../../../docs/rendered/estimation_examples/GPz_estimation_example_files/../../../docs/rendered/estimation_examples/GPz_estimation_example_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/GPz_estimation_example_files/../../../docs/rendered/estimation_examples/GPz_estimation_example_19_1.png

