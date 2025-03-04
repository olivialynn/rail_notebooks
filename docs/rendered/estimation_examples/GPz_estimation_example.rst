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
       1	-3.6048894e-01	 3.2561270e-01	-3.5076925e-01	 2.9898537e-01	[-3.0389794e-01]	 4.6344686e-01


.. parsed-literal::

       2	-2.8482741e-01	 3.1318997e-01	-2.5930125e-01	 2.8923447e-01	[-1.9106715e-01]	 2.3087096e-01


.. parsed-literal::

       3	-2.4009247e-01	 2.9244318e-01	-1.9676052e-01	 2.7505950e-01	[-1.2905969e-01]	 2.9066944e-01


.. parsed-literal::

       4	-1.9678880e-01	 2.7328005e-01	-1.4630530e-01	 2.6354593e-01	[-9.8302772e-02]	 2.9170108e-01


.. parsed-literal::

       5	-1.4723799e-01	 2.6028401e-01	-1.1762920e-01	 2.4874873e-01	[-5.3757983e-02]	 2.0962572e-01
       6	-8.3887062e-02	 2.5606546e-01	-5.6272818e-02	 2.4375061e-01	[-1.0424151e-02]	 1.7652988e-01


.. parsed-literal::

       7	-6.5149143e-02	 2.5195210e-01	-4.1523537e-02	 2.3916912e-01	[ 4.3610768e-03]	 1.8884134e-01


.. parsed-literal::

       8	-5.3647025e-02	 2.5004008e-01	-3.2917525e-02	 2.3725135e-01	[ 1.5600111e-02]	 2.0943046e-01


.. parsed-literal::

       9	-3.6495576e-02	 2.4677722e-01	-1.9164964e-02	 2.3478555e-01	[ 2.6721635e-02]	 2.1338201e-01


.. parsed-literal::

      10	-2.5655098e-02	 2.4463478e-01	-1.0289781e-02	 2.3384781e-01	[ 3.3833904e-02]	 2.0422602e-01
      11	-2.0025944e-02	 2.4365646e-01	-5.5567237e-03	 2.3331648e-01	[ 3.5739883e-02]	 1.9404221e-01


.. parsed-literal::

      12	-1.6234234e-02	 2.4290545e-01	-2.0777330e-03	 2.3307395e-01	[ 3.6364645e-02]	 1.7206979e-01


.. parsed-literal::

      13	-1.1577789e-02	 2.4196136e-01	 2.5152933e-03	 2.3260811e-01	[ 3.8892764e-02]	 2.2734833e-01


.. parsed-literal::

      14	 6.2819124e-02	 2.2891056e-01	 8.1868059e-02	 2.2198267e-01	[ 1.0492987e-01]	 3.1776500e-01


.. parsed-literal::

      15	 7.9360116e-02	 2.2633439e-01	 1.0059279e-01	 2.2030377e-01	[ 1.1284108e-01]	 3.0280733e-01


.. parsed-literal::

      16	 1.6656657e-01	 2.2025941e-01	 1.9131519e-01	 2.1736965e-01	[ 1.9924315e-01]	 2.1660924e-01


.. parsed-literal::

      17	 2.0689293e-01	 2.1645650e-01	 2.3876980e-01	 2.1289739e-01	[ 2.5826809e-01]	 2.0200300e-01


.. parsed-literal::

      18	 2.9354253e-01	 2.1343237e-01	 3.2527383e-01	 2.1155372e-01	[ 3.3459640e-01]	 2.1078897e-01


.. parsed-literal::

      19	 3.2186577e-01	 2.1057400e-01	 3.5370185e-01	 2.1103373e-01	[ 3.5922773e-01]	 2.1464896e-01
      20	 3.6368493e-01	 2.0736018e-01	 3.9712136e-01	 2.0826143e-01	[ 4.0346075e-01]	 1.9570208e-01


.. parsed-literal::

      21	 4.6913184e-01	 2.0641034e-01	 5.0237927e-01	 2.0828896e-01	[ 5.1010291e-01]	 2.1064711e-01


.. parsed-literal::

      22	 5.5733979e-01	 2.0521632e-01	 5.9396630e-01	 2.0719973e-01	[ 6.0031199e-01]	 2.1187687e-01


.. parsed-literal::

      23	 6.2753494e-01	 1.9546303e-01	 6.6671750e-01	 2.0081820e-01	[ 6.7398445e-01]	 2.0847726e-01


.. parsed-literal::

      24	 6.6539586e-01	 1.9191618e-01	 7.0392710e-01	 1.9550957e-01	[ 7.1461286e-01]	 2.1193433e-01


.. parsed-literal::

      25	 7.0005832e-01	 1.8978489e-01	 7.3825315e-01	 1.9090835e-01	[ 7.5338023e-01]	 2.0881224e-01


.. parsed-literal::

      26	 7.4736093e-01	 1.8476234e-01	 7.8601633e-01	 1.8468220e-01	[ 7.9563927e-01]	 2.1320057e-01
      27	 7.8878864e-01	 1.8926617e-01	 8.2795370e-01	 1.8656363e-01	[ 8.3267858e-01]	 1.8477106e-01


.. parsed-literal::

      28	 8.1918416e-01	 1.8953535e-01	 8.5874422e-01	 1.8614007e-01	[ 8.6375261e-01]	 2.1240783e-01


.. parsed-literal::

      29	 8.4775775e-01	 1.8998667e-01	 8.8815421e-01	 1.8759652e-01	[ 8.9193763e-01]	 2.0993543e-01


.. parsed-literal::

      30	 8.7967935e-01	 1.8593376e-01	 9.2095598e-01	 1.8463817e-01	[ 9.2818044e-01]	 2.1861815e-01
      31	 9.0591878e-01	 1.8089037e-01	 9.4699535e-01	 1.8312731e-01	[ 9.5172474e-01]	 1.7977500e-01


.. parsed-literal::

      32	 9.2705504e-01	 1.7531595e-01	 9.6898140e-01	 1.7605004e-01	[ 9.7167447e-01]	 2.0499539e-01


.. parsed-literal::

      33	 9.5204590e-01	 1.7134755e-01	 9.9422422e-01	 1.7279623e-01	[ 1.0012532e+00]	 2.1480274e-01
      34	 9.7697846e-01	 1.6693167e-01	 1.0199808e+00	 1.6774838e-01	[ 1.0216979e+00]	 1.9610596e-01


.. parsed-literal::

      35	 9.9989007e-01	 1.6430601e-01	 1.0443367e+00	 1.6471343e-01	[ 1.0419710e+00]	 2.1116662e-01
      36	 1.0150675e+00	 1.6291808e-01	 1.0605633e+00	 1.6183366e-01	[ 1.0573411e+00]	 1.9513178e-01


.. parsed-literal::

      37	 1.0287946e+00	 1.6141212e-01	 1.0744258e+00	 1.6106212e-01	[ 1.0678672e+00]	 2.0019245e-01
      38	 1.0430788e+00	 1.5936291e-01	 1.0888148e+00	 1.5983352e-01	[ 1.0823212e+00]	 1.7527986e-01


.. parsed-literal::

      39	 1.0626851e+00	 1.5662794e-01	 1.1087979e+00	 1.5713254e-01	[ 1.0981796e+00]	 1.8243289e-01


.. parsed-literal::

      40	 1.0730758e+00	 1.5608101e-01	 1.1198223e+00	 1.5771970e-01	[ 1.1063376e+00]	 2.1444988e-01
      41	 1.0855394e+00	 1.5537476e-01	 1.1316998e+00	 1.5687514e-01	[ 1.1142927e+00]	 1.7088604e-01


.. parsed-literal::

      42	 1.0950072e+00	 1.5435788e-01	 1.1412940e+00	 1.5475041e-01	[ 1.1209123e+00]	 2.0619774e-01


.. parsed-literal::

      43	 1.1102623e+00	 1.5265958e-01	 1.1568798e+00	 1.5105723e-01	[ 1.1293177e+00]	 2.1268630e-01


.. parsed-literal::

      44	 1.1272274e+00	 1.5045320e-01	 1.1739036e+00	 1.4629658e-01	[ 1.1471646e+00]	 2.0820999e-01
      45	 1.1412136e+00	 1.4921375e-01	 1.1879842e+00	 1.4400976e-01	[ 1.1572029e+00]	 1.9787979e-01


.. parsed-literal::

      46	 1.1521913e+00	 1.4837724e-01	 1.1990757e+00	 1.4286486e-01	[ 1.1697810e+00]	 2.0921659e-01


.. parsed-literal::

      47	 1.1653080e+00	 1.4756653e-01	 1.2128638e+00	 1.4108541e-01	[ 1.1824730e+00]	 2.0721221e-01


.. parsed-literal::

      48	 1.1737933e+00	 1.4649608e-01	 1.2219401e+00	 1.3993859e-01	[ 1.1963409e+00]	 2.0872736e-01


.. parsed-literal::

      49	 1.1819918e+00	 1.4606079e-01	 1.2300513e+00	 1.3973976e-01	[ 1.2009548e+00]	 2.0739222e-01


.. parsed-literal::

      50	 1.1929125e+00	 1.4504651e-01	 1.2412032e+00	 1.3892578e-01	[ 1.2042141e+00]	 2.1905279e-01
      51	 1.2028339e+00	 1.4409658e-01	 1.2512528e+00	 1.3840018e-01	[ 1.2105808e+00]	 1.7324519e-01


.. parsed-literal::

      52	 1.2212405e+00	 1.4245867e-01	 1.2703420e+00	 1.3711751e-01	[ 1.2205130e+00]	 1.8451786e-01


.. parsed-literal::

      53	 1.2239577e+00	 1.4132808e-01	 1.2735915e+00	 1.3711035e-01	[ 1.2248320e+00]	 2.0907950e-01
      54	 1.2414678e+00	 1.4062954e-01	 1.2902856e+00	 1.3637636e-01	[ 1.2401430e+00]	 1.9743538e-01


.. parsed-literal::

      55	 1.2481527e+00	 1.4030292e-01	 1.2971009e+00	 1.3637423e-01	[ 1.2423406e+00]	 1.8746066e-01


.. parsed-literal::

      56	 1.2583005e+00	 1.3949165e-01	 1.3075384e+00	 1.3638099e-01	[ 1.2470184e+00]	 2.1012306e-01


.. parsed-literal::

      57	 1.2684009e+00	 1.3847152e-01	 1.3179467e+00	 1.3586254e-01	[ 1.2523051e+00]	 2.0510077e-01


.. parsed-literal::

      58	 1.2799565e+00	 1.3678419e-01	 1.3299177e+00	 1.3473183e-01	[ 1.2626408e+00]	 2.0347548e-01
      59	 1.2887853e+00	 1.3633338e-01	 1.3386381e+00	 1.3406273e-01	[ 1.2711032e+00]	 1.9874978e-01


.. parsed-literal::

      60	 1.2961964e+00	 1.3602841e-01	 1.3460728e+00	 1.3352864e-01	[ 1.2790641e+00]	 2.1009326e-01


.. parsed-literal::

      61	 1.3032255e+00	 1.3567622e-01	 1.3532902e+00	 1.3351982e-01	[ 1.2840882e+00]	 2.1308422e-01


.. parsed-literal::

      62	 1.3111411e+00	 1.3496192e-01	 1.3612047e+00	 1.3286232e-01	[ 1.2961432e+00]	 2.0878029e-01
      63	 1.3197163e+00	 1.3437346e-01	 1.3699262e+00	 1.3277678e-01	[ 1.3016827e+00]	 2.0303011e-01


.. parsed-literal::

      64	 1.3295977e+00	 1.3363832e-01	 1.3800573e+00	 1.3242494e-01	[ 1.3117255e+00]	 1.9969392e-01


.. parsed-literal::

      65	 1.3355673e+00	 1.3334469e-01	 1.3862980e+00	 1.3232059e-01	  1.3080570e+00 	 2.0363140e-01
      66	 1.3429703e+00	 1.3323683e-01	 1.3935217e+00	 1.3209204e-01	[ 1.3163285e+00]	 1.9795847e-01


.. parsed-literal::

      67	 1.3494609e+00	 1.3288207e-01	 1.4000293e+00	 1.3178431e-01	[ 1.3207807e+00]	 2.0461988e-01
      68	 1.3556949e+00	 1.3276810e-01	 1.4064414e+00	 1.3161521e-01	[ 1.3209831e+00]	 1.8409133e-01


.. parsed-literal::

      69	 1.3618917e+00	 1.3181385e-01	 1.4130682e+00	 1.3104683e-01	  1.3209354e+00 	 2.0929980e-01


.. parsed-literal::

      70	 1.3704527e+00	 1.3145785e-01	 1.4215060e+00	 1.3082396e-01	[ 1.3287893e+00]	 2.0394492e-01


.. parsed-literal::

      71	 1.3764073e+00	 1.3079757e-01	 1.4275861e+00	 1.3059609e-01	[ 1.3324649e+00]	 2.0679045e-01
      72	 1.3811141e+00	 1.3011369e-01	 1.4326936e+00	 1.3041757e-01	  1.3304594e+00 	 1.7674994e-01


.. parsed-literal::

      73	 1.3862941e+00	 1.2944559e-01	 1.4379157e+00	 1.3006722e-01	[ 1.3342305e+00]	 2.0675826e-01


.. parsed-literal::

      74	 1.3920979e+00	 1.2888431e-01	 1.4437843e+00	 1.2975902e-01	[ 1.3358648e+00]	 2.1016932e-01


.. parsed-literal::

      75	 1.3991871e+00	 1.2839932e-01	 1.4508987e+00	 1.2943408e-01	  1.3352194e+00 	 2.1033096e-01
      76	 1.4050119e+00	 1.2767925e-01	 1.4568110e+00	 1.2901894e-01	  1.3290880e+00 	 1.8038774e-01


.. parsed-literal::

      77	 1.4107277e+00	 1.2762584e-01	 1.4623146e+00	 1.2876521e-01	  1.3358373e+00 	 2.1675038e-01


.. parsed-literal::

      78	 1.4143524e+00	 1.2747190e-01	 1.4659286e+00	 1.2853147e-01	[ 1.3367612e+00]	 2.0891404e-01


.. parsed-literal::

      79	 1.4181101e+00	 1.2728296e-01	 1.4697912e+00	 1.2814644e-01	[ 1.3397190e+00]	 2.0780993e-01
      80	 1.4220089e+00	 1.2707248e-01	 1.4737567e+00	 1.2787430e-01	[ 1.3398216e+00]	 2.0168853e-01


.. parsed-literal::

      81	 1.4262932e+00	 1.2682654e-01	 1.4781154e+00	 1.2771846e-01	  1.3387162e+00 	 1.7169952e-01
      82	 1.4324113e+00	 1.2650388e-01	 1.4844047e+00	 1.2717507e-01	[ 1.3419002e+00]	 1.9869256e-01


.. parsed-literal::

      83	 1.4361360e+00	 1.2639799e-01	 1.4880961e+00	 1.2727304e-01	  1.3391131e+00 	 1.9972944e-01


.. parsed-literal::

      84	 1.4386925e+00	 1.2624876e-01	 1.4905804e+00	 1.2717530e-01	  1.3411748e+00 	 2.1301866e-01


.. parsed-literal::

      85	 1.4426423e+00	 1.2608763e-01	 1.4946058e+00	 1.2690682e-01	  1.3417323e+00 	 2.1149683e-01
      86	 1.4464543e+00	 1.2591530e-01	 1.4986066e+00	 1.2660612e-01	  1.3378970e+00 	 1.9690013e-01


.. parsed-literal::

      87	 1.4505987e+00	 1.2569753e-01	 1.5030450e+00	 1.2613291e-01	  1.3392751e+00 	 2.1153212e-01


.. parsed-literal::

      88	 1.4546409e+00	 1.2551888e-01	 1.5071232e+00	 1.2574133e-01	  1.3409331e+00 	 2.0200706e-01


.. parsed-literal::

      89	 1.4577655e+00	 1.2526693e-01	 1.5101910e+00	 1.2555530e-01	[ 1.3440181e+00]	 2.0815825e-01


.. parsed-literal::

      90	 1.4609699e+00	 1.2503877e-01	 1.5133672e+00	 1.2519394e-01	[ 1.3492315e+00]	 2.0320582e-01


.. parsed-literal::

      91	 1.4657997e+00	 1.2455494e-01	 1.5181810e+00	 1.2448037e-01	[ 1.3537617e+00]	 2.0571852e-01
      92	 1.4688417e+00	 1.2446266e-01	 1.5212411e+00	 1.2421240e-01	  1.3535273e+00 	 1.8323493e-01


.. parsed-literal::

      93	 1.4710541e+00	 1.2441427e-01	 1.5233792e+00	 1.2404709e-01	[ 1.3582318e+00]	 2.1014762e-01
      94	 1.4732549e+00	 1.2437896e-01	 1.5255635e+00	 1.2396789e-01	[ 1.3599724e+00]	 1.7764544e-01


.. parsed-literal::

      95	 1.4764772e+00	 1.2424415e-01	 1.5288495e+00	 1.2387070e-01	[ 1.3621195e+00]	 2.0970726e-01
      96	 1.4778750e+00	 1.2415319e-01	 1.5305320e+00	 1.2378640e-01	  1.3542592e+00 	 1.9973540e-01


.. parsed-literal::

      97	 1.4825829e+00	 1.2400693e-01	 1.5351482e+00	 1.2366179e-01	  1.3619390e+00 	 2.2869182e-01


.. parsed-literal::

      98	 1.4841010e+00	 1.2390200e-01	 1.5366466e+00	 1.2355051e-01	[ 1.3636021e+00]	 2.0739365e-01
      99	 1.4867385e+00	 1.2375529e-01	 1.5393531e+00	 1.2327085e-01	[ 1.3645801e+00]	 1.7419219e-01


.. parsed-literal::

     100	 1.4885714e+00	 1.2355666e-01	 1.5413268e+00	 1.2278665e-01	[ 1.3649658e+00]	 2.0651150e-01
     101	 1.4912677e+00	 1.2352933e-01	 1.5439960e+00	 1.2264924e-01	[ 1.3671934e+00]	 2.0184350e-01


.. parsed-literal::

     102	 1.4930785e+00	 1.2350103e-01	 1.5458264e+00	 1.2250657e-01	[ 1.3682342e+00]	 1.7695475e-01


.. parsed-literal::

     103	 1.4949619e+00	 1.2343698e-01	 1.5477578e+00	 1.2233619e-01	[ 1.3687899e+00]	 2.0989227e-01


.. parsed-literal::

     104	 1.4984813e+00	 1.2318107e-01	 1.5514168e+00	 1.2202386e-01	  1.3680517e+00 	 2.0834708e-01


.. parsed-literal::

     105	 1.5005157e+00	 1.2309578e-01	 1.5535276e+00	 1.2174577e-01	  1.3672999e+00 	 3.2347536e-01
     106	 1.5025506e+00	 1.2286536e-01	 1.5555588e+00	 1.2163665e-01	  1.3674115e+00 	 2.0035505e-01


.. parsed-literal::

     107	 1.5047709e+00	 1.2260881e-01	 1.5577476e+00	 1.2151977e-01	  1.3651218e+00 	 2.1422911e-01
     108	 1.5068786e+00	 1.2224425e-01	 1.5598084e+00	 1.2135770e-01	  1.3644889e+00 	 1.9719911e-01


.. parsed-literal::

     109	 1.5090645e+00	 1.2213754e-01	 1.5619379e+00	 1.2133296e-01	  1.3598309e+00 	 2.0776916e-01


.. parsed-literal::

     110	 1.5109519e+00	 1.2208787e-01	 1.5637982e+00	 1.2128076e-01	  1.3573667e+00 	 2.1252704e-01
     111	 1.5131042e+00	 1.2192820e-01	 1.5659591e+00	 1.2122167e-01	  1.3560183e+00 	 1.9464993e-01


.. parsed-literal::

     112	 1.5155648e+00	 1.2160946e-01	 1.5685393e+00	 1.2111991e-01	  1.3507006e+00 	 2.1465993e-01
     113	 1.5179395e+00	 1.2125934e-01	 1.5709408e+00	 1.2109277e-01	  1.3534313e+00 	 1.7055392e-01


.. parsed-literal::

     114	 1.5195423e+00	 1.2111442e-01	 1.5725432e+00	 1.2109456e-01	  1.3538113e+00 	 1.9459414e-01
     115	 1.5209908e+00	 1.2094297e-01	 1.5740165e+00	 1.2105496e-01	  1.3540662e+00 	 1.7155385e-01


.. parsed-literal::

     116	 1.5225848e+00	 1.2077672e-01	 1.5756586e+00	 1.2108895e-01	  1.3511258e+00 	 2.0349264e-01


.. parsed-literal::

     117	 1.5244868e+00	 1.2069638e-01	 1.5775912e+00	 1.2099705e-01	  1.3512453e+00 	 2.2002578e-01
     118	 1.5258496e+00	 1.2066271e-01	 1.5789504e+00	 1.2098756e-01	  1.3499889e+00 	 1.8338990e-01


.. parsed-literal::

     119	 1.5278497e+00	 1.2061539e-01	 1.5809753e+00	 1.2097833e-01	  1.3483408e+00 	 2.1323919e-01


.. parsed-literal::

     120	 1.5299418e+00	 1.2047683e-01	 1.5831442e+00	 1.2098073e-01	  1.3439903e+00 	 2.1010423e-01


.. parsed-literal::

     121	 1.5318765e+00	 1.2036747e-01	 1.5851373e+00	 1.2111852e-01	  1.3396469e+00 	 2.1110415e-01
     122	 1.5331894e+00	 1.2029803e-01	 1.5864573e+00	 1.2103702e-01	  1.3428228e+00 	 1.8812990e-01


.. parsed-literal::

     123	 1.5343326e+00	 1.2023600e-01	 1.5875959e+00	 1.2098591e-01	  1.3416528e+00 	 2.1485543e-01
     124	 1.5362235e+00	 1.2024029e-01	 1.5894603e+00	 1.2086108e-01	  1.3378656e+00 	 1.8930507e-01


.. parsed-literal::

     125	 1.5373039e+00	 1.2026386e-01	 1.5905591e+00	 1.2069937e-01	  1.3346078e+00 	 2.0345116e-01


.. parsed-literal::

     126	 1.5385660e+00	 1.2030332e-01	 1.5917737e+00	 1.2068177e-01	  1.3325310e+00 	 2.0472455e-01


.. parsed-literal::

     127	 1.5394851e+00	 1.2035060e-01	 1.5926747e+00	 1.2067884e-01	  1.3314434e+00 	 2.1449399e-01


.. parsed-literal::

     128	 1.5407911e+00	 1.2038577e-01	 1.5939820e+00	 1.2068987e-01	  1.3302480e+00 	 2.2143245e-01
     129	 1.5434322e+00	 1.2040792e-01	 1.5966519e+00	 1.2074791e-01	  1.3295720e+00 	 1.9700408e-01


.. parsed-literal::

     130	 1.5446421e+00	 1.2036563e-01	 1.5979068e+00	 1.2088780e-01	  1.3266519e+00 	 3.0999708e-01


.. parsed-literal::

     131	 1.5462687e+00	 1.2034682e-01	 1.5995409e+00	 1.2095230e-01	  1.3304659e+00 	 2.2110891e-01


.. parsed-literal::

     132	 1.5475025e+00	 1.2025237e-01	 1.6007806e+00	 1.2099630e-01	  1.3325411e+00 	 2.1768260e-01
     133	 1.5489432e+00	 1.2011815e-01	 1.6022472e+00	 1.2112939e-01	  1.3317953e+00 	 2.0015907e-01


.. parsed-literal::

     134	 1.5505786e+00	 1.1997151e-01	 1.6039307e+00	 1.2121836e-01	  1.3270864e+00 	 2.0554757e-01


.. parsed-literal::

     135	 1.5518855e+00	 1.1983475e-01	 1.6052866e+00	 1.2139289e-01	  1.3141889e+00 	 2.2110677e-01


.. parsed-literal::

     136	 1.5530938e+00	 1.1980172e-01	 1.6065065e+00	 1.2143723e-01	  1.3103376e+00 	 2.2155809e-01
     137	 1.5541671e+00	 1.1973199e-01	 1.6075967e+00	 1.2148637e-01	  1.3067231e+00 	 1.8520498e-01


.. parsed-literal::

     138	 1.5548599e+00	 1.1965798e-01	 1.6084162e+00	 1.2173210e-01	  1.2944859e+00 	 2.1229982e-01
     139	 1.5566908e+00	 1.1951423e-01	 1.6102118e+00	 1.2175887e-01	  1.2965215e+00 	 1.9020653e-01


.. parsed-literal::

     140	 1.5576455e+00	 1.1939359e-01	 1.6111985e+00	 1.2186642e-01	  1.2949161e+00 	 2.0753193e-01
     141	 1.5585058e+00	 1.1930655e-01	 1.6120973e+00	 1.2195559e-01	  1.2918823e+00 	 1.9894195e-01


.. parsed-literal::

     142	 1.5601160e+00	 1.1922749e-01	 1.6137379e+00	 1.2199857e-01	  1.2850731e+00 	 2.1070433e-01


.. parsed-literal::

     143	 1.5608381e+00	 1.1923419e-01	 1.6144619e+00	 1.2201693e-01	  1.2820446e+00 	 2.9081845e-01


.. parsed-literal::

     144	 1.5619846e+00	 1.1924661e-01	 1.6155993e+00	 1.2202350e-01	  1.2752824e+00 	 2.0391583e-01


.. parsed-literal::

     145	 1.5629497e+00	 1.1926938e-01	 1.6165413e+00	 1.2205625e-01	  1.2702463e+00 	 2.0737743e-01
     146	 1.5644241e+00	 1.1926128e-01	 1.6179967e+00	 1.2218829e-01	  1.2621365e+00 	 1.9825077e-01


.. parsed-literal::

     147	 1.5654035e+00	 1.1931861e-01	 1.6189982e+00	 1.2248321e-01	  1.2485495e+00 	 2.1789718e-01
     148	 1.5668023e+00	 1.1925880e-01	 1.6203765e+00	 1.2254928e-01	  1.2494651e+00 	 2.0594621e-01


.. parsed-literal::

     149	 1.5682189e+00	 1.1920095e-01	 1.6218074e+00	 1.2272304e-01	  1.2466365e+00 	 2.0820117e-01
     150	 1.5694891e+00	 1.1919567e-01	 1.6231062e+00	 1.2290031e-01	  1.2414947e+00 	 1.7547965e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.08 s, total: 2min 5s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f890f607c70>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10000 - 20000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20000 - 20449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.75 s, sys: 41.9 ms, total: 1.79 s
    Wall time: 565 ms


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

