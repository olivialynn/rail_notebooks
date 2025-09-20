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

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4928862e-01	 3.2237667e-01	-3.3956195e-01	 3.1306836e-01	[-3.2269930e-01]	 4.5805955e-01


.. parsed-literal::

       2	-2.7640659e-01	 3.1078435e-01	-2.5155245e-01	 3.0217229e-01	[-2.2643885e-01]	 2.3069763e-01


.. parsed-literal::

       3	-2.3345735e-01	 2.9121140e-01	-1.9172274e-01	 2.8248907e-01	[-1.5857219e-01]	 2.7733350e-01
       4	-1.9873321e-01	 2.6675853e-01	-1.5809558e-01	 2.5926124e-01	[-1.1665001e-01]	 1.9129801e-01


.. parsed-literal::

       5	-1.0888882e-01	 2.5776660e-01	-7.2874261e-02	 2.5205334e-01	[-4.6774009e-02]	 1.8374777e-01


.. parsed-literal::

       6	-7.4519272e-02	 2.5248282e-01	-4.2925604e-02	 2.4917338e-01	[-2.9586547e-02]	 2.0565009e-01
       7	-5.5930822e-02	 2.4952499e-01	-3.1115208e-02	 2.4509824e-01	[-1.3800214e-02]	 1.7396283e-01


.. parsed-literal::

       8	-4.4074548e-02	 2.4760835e-01	-2.3163167e-02	 2.4282429e-01	[-4.3849321e-03]	 1.8469024e-01
       9	-3.0655472e-02	 2.4517649e-01	-1.2837882e-02	 2.4062488e-01	[ 5.0687604e-03]	 1.7324972e-01


.. parsed-literal::

      10	-1.9901349e-02	 2.4319621e-01	-4.4711384e-03	 2.3973830e-01	[ 1.0316508e-02]	 1.9682384e-01


.. parsed-literal::

      11	-1.4519917e-02	 2.4232673e-01	-3.4830818e-04	 2.3940288e-01	[ 1.1621874e-02]	 2.0210648e-01
      12	-1.1821016e-02	 2.4181714e-01	 2.1686389e-03	 2.3888651e-01	[ 1.4995026e-02]	 2.0307183e-01


.. parsed-literal::

      13	-8.0498562e-03	 2.4106067e-01	 5.8742164e-03	 2.3775555e-01	[ 2.0686566e-02]	 2.0678401e-01
      14	-2.9550055e-03	 2.3989332e-01	 1.1776821e-02	 2.3665106e-01	[ 2.7350972e-02]	 1.9657755e-01


.. parsed-literal::

      15	 1.5129687e-01	 2.2472013e-01	 1.7591939e-01	 2.2245886e-01	[ 1.8890934e-01]	 3.2280111e-01


.. parsed-literal::

      16	 2.8040521e-01	 2.1381125e-01	 3.1158332e-01	 2.1140937e-01	[ 3.3310090e-01]	 2.0692420e-01


.. parsed-literal::

      17	 3.0492282e-01	 2.1238503e-01	 3.3950180e-01	 2.1029350e-01	[ 3.6801610e-01]	 2.1095753e-01
      18	 3.6277914e-01	 2.0779273e-01	 3.9688060e-01	 2.0542378e-01	[ 4.2900470e-01]	 1.9842839e-01


.. parsed-literal::

      19	 3.9784785e-01	 2.0340346e-01	 4.3151013e-01	 2.0208460e-01	[ 4.6103165e-01]	 1.8426347e-01


.. parsed-literal::

      20	 4.7143608e-01	 1.9817083e-01	 5.0501030e-01	 1.9745927e-01	[ 5.3231853e-01]	 2.1429253e-01


.. parsed-literal::

      21	 5.9964537e-01	 1.9436074e-01	 6.3549648e-01	 1.9162506e-01	[ 6.6933558e-01]	 2.1272707e-01
      22	 6.4442406e-01	 1.9747821e-01	 6.8472266e-01	 1.9289570e-01	[ 7.3994158e-01]	 1.7723083e-01


.. parsed-literal::

      23	 6.8691115e-01	 1.8927312e-01	 7.2389188e-01	 1.8477464e-01	[ 7.5828792e-01]	 1.8631721e-01
      24	 7.0549108e-01	 1.8982048e-01	 7.4302260e-01	 1.8520400e-01	[ 7.7374564e-01]	 1.9780827e-01


.. parsed-literal::

      25	 7.3404353e-01	 1.9319462e-01	 7.7346275e-01	 1.9065979e-01	[ 7.8971284e-01]	 2.1051645e-01


.. parsed-literal::

      26	 7.8006001e-01	 1.8794170e-01	 8.1926143e-01	 1.8530790e-01	[ 8.3111905e-01]	 2.0713162e-01
      27	 8.0873117e-01	 1.8623958e-01	 8.4791830e-01	 1.8384583e-01	[ 8.5793700e-01]	 1.9569898e-01


.. parsed-literal::

      28	 8.3766325e-01	 1.8666795e-01	 8.7686164e-01	 1.8503809e-01	[ 8.8606978e-01]	 2.1074963e-01


.. parsed-literal::

      29	 8.6139367e-01	 1.8637656e-01	 9.0100475e-01	 1.8555409e-01	[ 9.0917144e-01]	 2.0794749e-01
      30	 8.8900798e-01	 1.8464472e-01	 9.2955291e-01	 1.8396779e-01	[ 9.3753138e-01]	 1.9854379e-01


.. parsed-literal::

      31	 9.1402907e-01	 1.8077919e-01	 9.5614718e-01	 1.8098600e-01	[ 9.5603277e-01]	 1.8130708e-01
      32	 9.3067055e-01	 1.7610105e-01	 9.7347558e-01	 1.7600436e-01	[ 9.7265349e-01]	 1.9663095e-01


.. parsed-literal::

      33	 9.4827721e-01	 1.7333494e-01	 9.9048847e-01	 1.7248710e-01	[ 9.9437217e-01]	 2.0091867e-01


.. parsed-literal::

      34	 9.7171513e-01	 1.6876739e-01	 1.0140879e+00	 1.6589293e-01	[ 1.0126108e+00]	 2.1325278e-01
      35	 9.8854128e-01	 1.6679910e-01	 1.0316440e+00	 1.6255704e-01	[ 1.0290182e+00]	 1.9932294e-01


.. parsed-literal::

      36	 1.0092353e+00	 1.6383280e-01	 1.0537490e+00	 1.5797035e-01	[ 1.0472051e+00]	 1.9549680e-01


.. parsed-literal::

      37	 1.0199384e+00	 1.6227776e-01	 1.0652815e+00	 1.5561293e-01	[ 1.0474568e+00]	 2.1701169e-01
      38	 1.0298841e+00	 1.6111751e-01	 1.0749450e+00	 1.5448244e-01	[ 1.0592070e+00]	 1.9388199e-01


.. parsed-literal::

      39	 1.0410473e+00	 1.5942613e-01	 1.0862295e+00	 1.5241275e-01	[ 1.0716981e+00]	 2.0910168e-01
      40	 1.0505502e+00	 1.5810784e-01	 1.0960873e+00	 1.5098105e-01	[ 1.0767603e+00]	 1.7970562e-01


.. parsed-literal::

      41	 1.0709941e+00	 1.5448353e-01	 1.1172472e+00	 1.4703547e-01	[ 1.0898620e+00]	 2.0768595e-01
      42	 1.0798990e+00	 1.5325274e-01	 1.1265243e+00	 1.4636419e-01	[ 1.0964104e+00]	 1.9152164e-01


.. parsed-literal::

      43	 1.0892899e+00	 1.5257237e-01	 1.1356136e+00	 1.4574379e-01	[ 1.1094583e+00]	 2.0275950e-01


.. parsed-literal::

      44	 1.0981511e+00	 1.5163670e-01	 1.1445504e+00	 1.4488264e-01	[ 1.1218600e+00]	 2.1042395e-01


.. parsed-literal::

      45	 1.1068994e+00	 1.5121259e-01	 1.1535436e+00	 1.4434435e-01	[ 1.1365676e+00]	 2.1215701e-01
      46	 1.1187283e+00	 1.5110429e-01	 1.1656013e+00	 1.4390425e-01	[ 1.1562834e+00]	 1.9742775e-01


.. parsed-literal::

      47	 1.1274513e+00	 1.5006481e-01	 1.1745736e+00	 1.4278383e-01	[ 1.1680631e+00]	 1.9881773e-01
      48	 1.1347582e+00	 1.4902210e-01	 1.1821008e+00	 1.4192507e-01	[ 1.1718717e+00]	 1.9795108e-01


.. parsed-literal::

      49	 1.1450239e+00	 1.4721958e-01	 1.1926979e+00	 1.4064209e-01	[ 1.1759003e+00]	 2.0647478e-01


.. parsed-literal::

      50	 1.1568555e+00	 1.4491296e-01	 1.2052126e+00	 1.3923669e-01	[ 1.1769018e+00]	 2.1163344e-01


.. parsed-literal::

      51	 1.1685697e+00	 1.4373265e-01	 1.2166869e+00	 1.3844456e-01	[ 1.1888456e+00]	 2.1533751e-01


.. parsed-literal::

      52	 1.1756908e+00	 1.4345955e-01	 1.2236698e+00	 1.3814797e-01	[ 1.1976538e+00]	 2.1522570e-01
      53	 1.1884860e+00	 1.4319712e-01	 1.2367714e+00	 1.3774122e-01	[ 1.2115086e+00]	 1.8441749e-01


.. parsed-literal::

      54	 1.1990375e+00	 1.4377512e-01	 1.2472337e+00	 1.3842157e-01	[ 1.2239203e+00]	 2.0390081e-01
      55	 1.2088142e+00	 1.4321227e-01	 1.2572965e+00	 1.3775034e-01	[ 1.2309502e+00]	 1.8493319e-01


.. parsed-literal::

      56	 1.2170739e+00	 1.4229568e-01	 1.2659306e+00	 1.3705781e-01	[ 1.2352850e+00]	 2.0967388e-01


.. parsed-literal::

      57	 1.2272324e+00	 1.4159302e-01	 1.2765706e+00	 1.3625280e-01	[ 1.2417301e+00]	 2.1455574e-01


.. parsed-literal::

      58	 1.2365925e+00	 1.4084184e-01	 1.2863694e+00	 1.3593386e-01	[ 1.2449594e+00]	 2.0537090e-01


.. parsed-literal::

      59	 1.2437412e+00	 1.4048349e-01	 1.2934669e+00	 1.3545833e-01	[ 1.2540802e+00]	 2.1648145e-01
      60	 1.2515057e+00	 1.4033637e-01	 1.3012447e+00	 1.3526925e-01	[ 1.2655586e+00]	 1.9879794e-01


.. parsed-literal::

      61	 1.2589753e+00	 1.3984280e-01	 1.3088905e+00	 1.3416145e-01	[ 1.2787185e+00]	 2.1796465e-01


.. parsed-literal::

      62	 1.2663927e+00	 1.3953002e-01	 1.3163033e+00	 1.3374061e-01	[ 1.2889494e+00]	 2.1612048e-01
      63	 1.2751504e+00	 1.3935423e-01	 1.3254006e+00	 1.3351739e-01	[ 1.2933743e+00]	 1.9977927e-01


.. parsed-literal::

      64	 1.2809454e+00	 1.3915984e-01	 1.3312936e+00	 1.3357525e-01	[ 1.3041675e+00]	 2.1325755e-01
      65	 1.2865548e+00	 1.3890934e-01	 1.3367293e+00	 1.3381756e-01	[ 1.3075512e+00]	 1.8305326e-01


.. parsed-literal::

      66	 1.2934780e+00	 1.3850060e-01	 1.3438939e+00	 1.3397968e-01	[ 1.3119988e+00]	 2.1346068e-01
      67	 1.2994526e+00	 1.3806302e-01	 1.3499555e+00	 1.3427138e-01	[ 1.3150247e+00]	 1.9867706e-01


.. parsed-literal::

      68	 1.3063861e+00	 1.3749913e-01	 1.3569731e+00	 1.3416922e-01	[ 1.3180423e+00]	 2.1415949e-01


.. parsed-literal::

      69	 1.3142997e+00	 1.3706374e-01	 1.3651907e+00	 1.3420057e-01	[ 1.3220265e+00]	 2.1558881e-01
      70	 1.3220940e+00	 1.3650292e-01	 1.3731973e+00	 1.3415972e-01	[ 1.3267675e+00]	 1.8442106e-01


.. parsed-literal::

      71	 1.3285536e+00	 1.3624880e-01	 1.3796026e+00	 1.3387216e-01	[ 1.3312640e+00]	 2.0877051e-01


.. parsed-literal::

      72	 1.3348682e+00	 1.3558274e-01	 1.3859783e+00	 1.3363348e-01	[ 1.3374068e+00]	 2.1054769e-01


.. parsed-literal::

      73	 1.3411039e+00	 1.3524767e-01	 1.3923061e+00	 1.3342016e-01	[ 1.3443601e+00]	 2.0379972e-01


.. parsed-literal::

      74	 1.3493444e+00	 1.3470414e-01	 1.4007243e+00	 1.3347550e-01	[ 1.3472902e+00]	 2.1416450e-01
      75	 1.3552704e+00	 1.3447569e-01	 1.4067502e+00	 1.3302734e-01	  1.3472005e+00 	 1.9410825e-01


.. parsed-literal::

      76	 1.3607493e+00	 1.3447798e-01	 1.4122799e+00	 1.3307014e-01	[ 1.3511298e+00]	 2.0274854e-01


.. parsed-literal::

      77	 1.3653230e+00	 1.3440842e-01	 1.4170367e+00	 1.3299802e-01	[ 1.3515080e+00]	 2.1638298e-01


.. parsed-literal::

      78	 1.3713704e+00	 1.3427923e-01	 1.4232318e+00	 1.3296588e-01	[ 1.3516977e+00]	 2.0556426e-01


.. parsed-literal::

      79	 1.3772415e+00	 1.3414163e-01	 1.4293870e+00	 1.3246808e-01	[ 1.3558225e+00]	 2.0931482e-01
      80	 1.3837549e+00	 1.3391199e-01	 1.4358482e+00	 1.3298365e-01	[ 1.3601116e+00]	 1.7631745e-01


.. parsed-literal::

      81	 1.3881947e+00	 1.3391787e-01	 1.4401721e+00	 1.3337414e-01	[ 1.3659459e+00]	 2.0407438e-01
      82	 1.3931202e+00	 1.3391318e-01	 1.4452973e+00	 1.3381468e-01	[ 1.3705462e+00]	 1.7956495e-01


.. parsed-literal::

      83	 1.3975330e+00	 1.3381366e-01	 1.4498434e+00	 1.3391048e-01	  1.3648486e+00 	 1.9958949e-01


.. parsed-literal::

      84	 1.4013934e+00	 1.3367027e-01	 1.4536174e+00	 1.3370172e-01	  1.3654473e+00 	 2.0873475e-01


.. parsed-literal::

      85	 1.4050663e+00	 1.3348615e-01	 1.4573781e+00	 1.3335699e-01	  1.3606169e+00 	 2.0993161e-01


.. parsed-literal::

      86	 1.4087720e+00	 1.3301137e-01	 1.4612119e+00	 1.3291127e-01	  1.3615144e+00 	 2.1851420e-01
      87	 1.4137101e+00	 1.3258303e-01	 1.4662281e+00	 1.3264462e-01	  1.3637410e+00 	 1.9632292e-01


.. parsed-literal::

      88	 1.4171661e+00	 1.3217738e-01	 1.4697051e+00	 1.3264511e-01	[ 1.3711644e+00]	 2.0849037e-01


.. parsed-literal::

      89	 1.4205897e+00	 1.3205342e-01	 1.4730680e+00	 1.3275445e-01	[ 1.3782152e+00]	 2.0678043e-01


.. parsed-literal::

      90	 1.4245559e+00	 1.3183565e-01	 1.4769787e+00	 1.3259055e-01	[ 1.3859355e+00]	 2.1264410e-01


.. parsed-literal::

      91	 1.4306397e+00	 1.3148719e-01	 1.4832128e+00	 1.3257239e-01	[ 1.3903814e+00]	 2.0939994e-01


.. parsed-literal::

      92	 1.4344855e+00	 1.3108789e-01	 1.4871497e+00	 1.3202953e-01	  1.3899889e+00 	 3.1649971e-01


.. parsed-literal::

      93	 1.4390935e+00	 1.3085604e-01	 1.4918031e+00	 1.3189280e-01	[ 1.3914297e+00]	 2.0506978e-01
      94	 1.4429827e+00	 1.3058761e-01	 1.4957494e+00	 1.3174705e-01	  1.3888365e+00 	 1.7644429e-01


.. parsed-literal::

      95	 1.4463778e+00	 1.3041470e-01	 1.4992701e+00	 1.3169226e-01	  1.3835127e+00 	 2.0760059e-01
      96	 1.4494092e+00	 1.3035533e-01	 1.5023339e+00	 1.3173830e-01	  1.3837463e+00 	 1.9966793e-01


.. parsed-literal::

      97	 1.4516313e+00	 1.3032834e-01	 1.5045276e+00	 1.3168040e-01	  1.3859878e+00 	 2.1664071e-01


.. parsed-literal::

      98	 1.4542156e+00	 1.3034060e-01	 1.5072177e+00	 1.3184814e-01	  1.3866313e+00 	 2.0113540e-01
      99	 1.4558309e+00	 1.3030871e-01	 1.5090769e+00	 1.3172521e-01	  1.3795606e+00 	 2.0042253e-01


.. parsed-literal::

     100	 1.4580642e+00	 1.3021718e-01	 1.5112249e+00	 1.3172131e-01	  1.3819105e+00 	 2.0008612e-01


.. parsed-literal::

     101	 1.4602345e+00	 1.3010837e-01	 1.5134379e+00	 1.3170302e-01	  1.3794905e+00 	 2.0584297e-01
     102	 1.4617575e+00	 1.3002430e-01	 1.5149939e+00	 1.3164093e-01	  1.3784823e+00 	 1.8927979e-01


.. parsed-literal::

     103	 1.4651601e+00	 1.2982858e-01	 1.5185769e+00	 1.3130950e-01	  1.3728799e+00 	 2.1661091e-01


.. parsed-literal::

     104	 1.4682393e+00	 1.2967582e-01	 1.5217991e+00	 1.3128189e-01	  1.3784637e+00 	 2.0706058e-01


.. parsed-literal::

     105	 1.4705258e+00	 1.2955476e-01	 1.5239798e+00	 1.3111377e-01	  1.3809172e+00 	 2.0118856e-01
     106	 1.4727927e+00	 1.2940865e-01	 1.5262480e+00	 1.3092304e-01	  1.3827330e+00 	 1.7934918e-01


.. parsed-literal::

     107	 1.4751294e+00	 1.2924096e-01	 1.5286203e+00	 1.3066955e-01	  1.3836442e+00 	 2.0083714e-01
     108	 1.4767460e+00	 1.2907849e-01	 1.5304903e+00	 1.3015037e-01	  1.3743268e+00 	 1.8292570e-01


.. parsed-literal::

     109	 1.4799567e+00	 1.2897789e-01	 1.5335131e+00	 1.3015997e-01	  1.3810418e+00 	 2.1884441e-01


.. parsed-literal::

     110	 1.4808520e+00	 1.2893850e-01	 1.5343856e+00	 1.3010620e-01	  1.3805773e+00 	 2.0981216e-01


.. parsed-literal::

     111	 1.4833445e+00	 1.2880437e-01	 1.5369252e+00	 1.2989542e-01	  1.3772931e+00 	 2.0448375e-01


.. parsed-literal::

     112	 1.4843186e+00	 1.2865951e-01	 1.5381397e+00	 1.2948419e-01	  1.3687919e+00 	 2.0188141e-01
     113	 1.4869497e+00	 1.2858661e-01	 1.5406673e+00	 1.2947513e-01	  1.3714899e+00 	 1.9798398e-01


.. parsed-literal::

     114	 1.4882988e+00	 1.2852248e-01	 1.5420307e+00	 1.2941673e-01	  1.3711681e+00 	 2.1186996e-01


.. parsed-literal::

     115	 1.4899255e+00	 1.2842131e-01	 1.5437280e+00	 1.2931709e-01	  1.3691545e+00 	 2.0499635e-01


.. parsed-literal::

     116	 1.4920938e+00	 1.2818718e-01	 1.5460138e+00	 1.2928309e-01	  1.3644679e+00 	 2.1948266e-01
     117	 1.4943389e+00	 1.2800761e-01	 1.5483678e+00	 1.2910868e-01	  1.3596695e+00 	 1.8654823e-01


.. parsed-literal::

     118	 1.4959523e+00	 1.2795704e-01	 1.5499340e+00	 1.2912536e-01	  1.3608206e+00 	 2.0662308e-01


.. parsed-literal::

     119	 1.4981777e+00	 1.2782201e-01	 1.5522187e+00	 1.2914747e-01	  1.3577819e+00 	 2.0113659e-01


.. parsed-literal::

     120	 1.4988284e+00	 1.2779408e-01	 1.5530107e+00	 1.2913017e-01	  1.3534632e+00 	 2.1268368e-01
     121	 1.5004069e+00	 1.2774149e-01	 1.5545127e+00	 1.2908590e-01	  1.3554068e+00 	 2.0439410e-01


.. parsed-literal::

     122	 1.5015093e+00	 1.2768520e-01	 1.5556539e+00	 1.2899668e-01	  1.3536215e+00 	 1.9820380e-01


.. parsed-literal::

     123	 1.5028322e+00	 1.2764188e-01	 1.5570538e+00	 1.2888842e-01	  1.3499372e+00 	 2.1128941e-01


.. parsed-literal::

     124	 1.5048979e+00	 1.2759162e-01	 1.5591746e+00	 1.2883951e-01	  1.3444668e+00 	 2.1051693e-01


.. parsed-literal::

     125	 1.5060824e+00	 1.2760683e-01	 1.5604015e+00	 1.2874352e-01	  1.3393132e+00 	 2.9909325e-01


.. parsed-literal::

     126	 1.5075817e+00	 1.2758141e-01	 1.5618817e+00	 1.2880784e-01	  1.3370590e+00 	 2.0213056e-01


.. parsed-literal::

     127	 1.5087858e+00	 1.2754279e-01	 1.5630468e+00	 1.2887271e-01	  1.3385638e+00 	 2.2081804e-01
     128	 1.5104966e+00	 1.2743819e-01	 1.5647500e+00	 1.2891256e-01	  1.3390493e+00 	 1.8538976e-01


.. parsed-literal::

     129	 1.5111879e+00	 1.2730509e-01	 1.5655037e+00	 1.2915147e-01	  1.3392306e+00 	 1.9954252e-01
     130	 1.5127229e+00	 1.2725282e-01	 1.5670012e+00	 1.2899343e-01	  1.3392701e+00 	 1.8069839e-01


.. parsed-literal::

     131	 1.5137043e+00	 1.2716044e-01	 1.5680360e+00	 1.2888845e-01	  1.3361098e+00 	 2.0872140e-01


.. parsed-literal::

     132	 1.5148593e+00	 1.2701122e-01	 1.5692776e+00	 1.2884146e-01	  1.3303182e+00 	 2.0812941e-01
     133	 1.5167398e+00	 1.2678992e-01	 1.5712665e+00	 1.2889872e-01	  1.3206532e+00 	 1.9994164e-01


.. parsed-literal::

     134	 1.5176721e+00	 1.2663857e-01	 1.5722579e+00	 1.2899235e-01	  1.3142761e+00 	 3.2406592e-01


.. parsed-literal::

     135	 1.5190696e+00	 1.2651098e-01	 1.5736715e+00	 1.2909318e-01	  1.3106274e+00 	 2.0847487e-01


.. parsed-literal::

     136	 1.5201989e+00	 1.2642095e-01	 1.5747987e+00	 1.2915599e-01	  1.3113830e+00 	 2.0440340e-01
     137	 1.5213537e+00	 1.2637896e-01	 1.5759577e+00	 1.2922167e-01	  1.3116097e+00 	 2.0178938e-01


.. parsed-literal::

     138	 1.5226251e+00	 1.2630512e-01	 1.5771990e+00	 1.2915514e-01	  1.3144152e+00 	 1.9966507e-01


.. parsed-literal::

     139	 1.5236133e+00	 1.2627609e-01	 1.5781934e+00	 1.2906216e-01	  1.3141509e+00 	 2.2163033e-01


.. parsed-literal::

     140	 1.5249814e+00	 1.2623448e-01	 1.5796270e+00	 1.2894287e-01	  1.3090171e+00 	 2.1908927e-01


.. parsed-literal::

     141	 1.5262004e+00	 1.2622468e-01	 1.5809338e+00	 1.2893795e-01	  1.2992196e+00 	 2.2102976e-01


.. parsed-literal::

     142	 1.5273362e+00	 1.2616377e-01	 1.5821865e+00	 1.2887694e-01	  1.2833627e+00 	 2.2394323e-01


.. parsed-literal::

     143	 1.5281488e+00	 1.2619851e-01	 1.5829546e+00	 1.2894224e-01	  1.2841973e+00 	 2.0471644e-01


.. parsed-literal::

     144	 1.5287911e+00	 1.2618597e-01	 1.5835908e+00	 1.2895062e-01	  1.2836292e+00 	 2.0622683e-01


.. parsed-literal::

     145	 1.5302112e+00	 1.2616571e-01	 1.5850229e+00	 1.2878074e-01	  1.2832354e+00 	 2.1023464e-01


.. parsed-literal::

     146	 1.5310737e+00	 1.2611502e-01	 1.5860048e+00	 1.2851298e-01	  1.2779098e+00 	 2.1044970e-01


.. parsed-literal::

     147	 1.5328276e+00	 1.2606722e-01	 1.5876609e+00	 1.2828124e-01	  1.2796254e+00 	 2.1887994e-01
     148	 1.5334689e+00	 1.2604266e-01	 1.5882887e+00	 1.2818213e-01	  1.2789297e+00 	 1.7996979e-01


.. parsed-literal::

     149	 1.5346892e+00	 1.2598505e-01	 1.5895269e+00	 1.2801310e-01	  1.2761903e+00 	 2.1398306e-01


.. parsed-literal::

     150	 1.5353525e+00	 1.2599209e-01	 1.5902329e+00	 1.2793014e-01	  1.2740190e+00 	 3.1219006e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.09 s, total: 2min 5s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fa66ce41900>



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


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.02 s, sys: 47 ms, total: 2.07 s
    Wall time: 617 ms


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




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


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




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

