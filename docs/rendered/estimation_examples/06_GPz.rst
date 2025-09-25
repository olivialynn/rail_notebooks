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
       1	-3.5260548e-01	 3.2335954e-01	-3.4286858e-01	 3.1006179e-01	[-3.1702320e-01]	 4.6212721e-01


.. parsed-literal::

       2	-2.8067852e-01	 3.1229580e-01	-2.5675795e-01	 3.0192539e-01	[-2.2021965e-01]	 2.3648024e-01


.. parsed-literal::

       3	-2.3599938e-01	 2.9162990e-01	-1.9432378e-01	 2.8027614e-01	[-1.4468353e-01]	 2.7958775e-01


.. parsed-literal::

       4	-2.0533041e-01	 2.6733178e-01	-1.6518471e-01	 2.5781269e-01	[-1.0845645e-01]	 2.1671700e-01


.. parsed-literal::

       5	-1.1295291e-01	 2.5914195e-01	-7.8359829e-02	 2.5396226e-01	[-4.3837259e-02]	 2.0641184e-01


.. parsed-literal::

       6	-7.7769355e-02	 2.5313502e-01	-4.6257882e-02	 2.4674121e-01	[-2.1862877e-02]	 2.0697069e-01


.. parsed-literal::

       7	-6.0091876e-02	 2.5065233e-01	-3.5768432e-02	 2.4480521e-01	[-1.0978235e-02]	 2.0828938e-01


.. parsed-literal::

       8	-4.6947014e-02	 2.4844332e-01	-2.6601928e-02	 2.4201509e-01	[ 1.0603556e-03]	 2.1657991e-01
       9	-3.2798166e-02	 2.4583039e-01	-1.5499479e-02	 2.3832117e-01	[ 1.6330002e-02]	 1.8844962e-01


.. parsed-literal::

      10	-2.4279770e-02	 2.4442560e-01	-8.9834775e-03	 2.3658617e-01	[ 2.1915580e-02]	 2.0374894e-01


.. parsed-literal::

      11	-1.7501816e-02	 2.4299024e-01	-2.8949807e-03	 2.3645170e-01	[ 2.4464542e-02]	 2.0819592e-01


.. parsed-literal::

      12	-1.4946845e-02	 2.4249214e-01	-5.1177119e-04	 2.3599195e-01	[ 2.6363312e-02]	 2.1971035e-01


.. parsed-literal::

      13	-1.0232860e-02	 2.4156705e-01	 4.0895411e-03	 2.3534729e-01	[ 3.0922077e-02]	 2.0852590e-01
      14	-1.0833077e-03	 2.3928577e-01	 1.4863143e-02	 2.3351572e-01	[ 3.9997302e-02]	 1.8518853e-01


.. parsed-literal::

      15	 3.3274758e-02	 2.2824307e-01	 5.0879088e-02	 2.2708760e-01	[ 6.3736405e-02]	 2.0640731e-01


.. parsed-literal::

      16	 9.6536747e-02	 2.2156890e-01	 1.1559579e-01	 2.2225743e-01	[ 1.2463496e-01]	 2.0731521e-01
      17	 1.8882699e-01	 2.2171359e-01	 2.2000053e-01	 2.2237275e-01	[ 2.3721108e-01]	 1.8238282e-01


.. parsed-literal::

      18	 2.6910446e-01	 2.1920612e-01	 2.9885473e-01	 2.1368280e-01	[ 2.9568377e-01]	 2.1219802e-01


.. parsed-literal::

      19	 3.0884436e-01	 2.1358954e-01	 3.4021846e-01	 2.0899600e-01	[ 3.3306972e-01]	 2.1084571e-01


.. parsed-literal::

      20	 3.5721960e-01	 2.0992427e-01	 3.8763141e-01	 2.0575703e-01	[ 3.8112302e-01]	 2.0348716e-01


.. parsed-literal::

      21	 4.2066567e-01	 2.0683250e-01	 4.5254752e-01	 2.0309599e-01	[ 4.4128312e-01]	 2.1173716e-01
      22	 5.0622172e-01	 2.0499025e-01	 5.4193954e-01	 2.0241839e-01	[ 5.2367959e-01]	 2.0142484e-01


.. parsed-literal::

      23	 5.6520387e-01	 2.0156682e-01	 6.0389003e-01	 1.9931615e-01	[ 5.8699592e-01]	 2.0900702e-01
      24	 6.1000612e-01	 1.9795356e-01	 6.4875187e-01	 1.9664140e-01	[ 6.3558650e-01]	 1.7943716e-01


.. parsed-literal::

      25	 6.5086802e-01	 1.9467155e-01	 6.8912875e-01	 1.9120888e-01	[ 6.8101971e-01]	 2.1463633e-01
      26	 6.8542100e-01	 1.9468562e-01	 7.2223190e-01	 1.9267981e-01	[ 7.0867326e-01]	 1.8061447e-01


.. parsed-literal::

      27	 7.1079671e-01	 1.9823326e-01	 7.4554825e-01	 1.9546078e-01	[ 7.3487541e-01]	 2.1922016e-01
      28	 7.3757189e-01	 1.9866200e-01	 7.7314508e-01	 1.9523256e-01	[ 7.6355171e-01]	 1.8559957e-01


.. parsed-literal::

      29	 7.4575702e-01	 2.0683877e-01	 7.8198921e-01	 2.0057920e-01	[ 7.8542384e-01]	 2.0054865e-01


.. parsed-literal::

      30	 7.5668103e-01	 2.1168203e-01	 7.9837401e-01	 2.0340479e-01	[ 7.8869205e-01]	 2.1257377e-01


.. parsed-literal::

      31	 7.8341764e-01	 2.0969134e-01	 8.2205469e-01	 2.0344121e-01	[ 8.0385293e-01]	 2.1603489e-01


.. parsed-literal::

      32	 8.1097731e-01	 2.0461104e-01	 8.4944467e-01	 2.0153806e-01	[ 8.3621911e-01]	 2.1108651e-01


.. parsed-literal::

      33	 8.1777772e-01	 2.0312051e-01	 8.5652895e-01	 1.9927729e-01	[ 8.4617096e-01]	 2.0470023e-01
      34	 8.4241371e-01	 2.0284150e-01	 8.8153270e-01	 2.0095692e-01	[ 8.6757106e-01]	 1.8557310e-01


.. parsed-literal::

      35	 8.5876120e-01	 2.0401177e-01	 8.9845746e-01	 2.0223426e-01	[ 8.7976543e-01]	 1.9514060e-01


.. parsed-literal::

      36	 8.8227621e-01	 2.0719610e-01	 9.2342002e-01	 2.0593891e-01	[ 8.9318708e-01]	 2.0971942e-01


.. parsed-literal::

      37	 9.0264877e-01	 2.0962879e-01	 9.4495036e-01	 2.0444492e-01	[ 9.0586827e-01]	 2.0757818e-01


.. parsed-literal::

      38	 9.1577727e-01	 2.0697651e-01	 9.5780459e-01	 2.0051272e-01	[ 9.1688580e-01]	 2.0574212e-01


.. parsed-literal::

      39	 9.4257321e-01	 2.0084636e-01	 9.8495236e-01	 1.9457024e-01	[ 9.4215795e-01]	 2.0944309e-01
      40	 9.5970670e-01	 1.9716774e-01	 1.0035322e+00	 1.8841563e-01	[ 9.4887809e-01]	 1.8615127e-01


.. parsed-literal::

      41	 9.7521810e-01	 1.9607688e-01	 1.0192963e+00	 1.8848027e-01	[ 9.6240268e-01]	 1.6899085e-01


.. parsed-literal::

      42	 9.8356663e-01	 1.9558439e-01	 1.0281246e+00	 1.8723347e-01	[ 9.6822214e-01]	 2.2132206e-01


.. parsed-literal::

      43	 9.9867676e-01	 1.9370588e-01	 1.0440399e+00	 1.8478680e-01	[ 9.7388734e-01]	 2.1014953e-01


.. parsed-literal::

      44	 1.0115501e+00	 1.9236286e-01	 1.0574565e+00	 1.8250765e-01	[ 9.8162769e-01]	 2.1412373e-01
      45	 1.0237430e+00	 1.8966240e-01	 1.0696529e+00	 1.8049311e-01	[ 9.9184540e-01]	 1.6972327e-01


.. parsed-literal::

      46	 1.0345432e+00	 1.8869078e-01	 1.0804718e+00	 1.7913830e-01	[ 1.0021270e+00]	 2.0296621e-01
      47	 1.0470023e+00	 1.8689787e-01	 1.0934317e+00	 1.7729068e-01	[ 1.0080802e+00]	 1.9868016e-01


.. parsed-literal::

      48	 1.0578596e+00	 1.8392914e-01	 1.1045571e+00	 1.7426625e-01	[ 1.0138163e+00]	 2.0113444e-01
      49	 1.0687526e+00	 1.8233236e-01	 1.1154871e+00	 1.7259704e-01	[ 1.0236586e+00]	 1.9953442e-01


.. parsed-literal::

      50	 1.0827108e+00	 1.8012397e-01	 1.1297088e+00	 1.7095041e-01	[ 1.0348640e+00]	 2.1088028e-01
      51	 1.0960521e+00	 1.7789285e-01	 1.1435463e+00	 1.6864391e-01	[ 1.0474784e+00]	 1.7164230e-01


.. parsed-literal::

      52	 1.1087553e+00	 1.7384275e-01	 1.1564834e+00	 1.6505055e-01	[ 1.0653925e+00]	 2.0369244e-01
      53	 1.1200454e+00	 1.7343251e-01	 1.1678939e+00	 1.6382578e-01	[ 1.0739314e+00]	 2.0147395e-01


.. parsed-literal::

      54	 1.1283366e+00	 1.7236084e-01	 1.1763580e+00	 1.6272861e-01	[ 1.0796756e+00]	 2.1185017e-01


.. parsed-literal::

      55	 1.1432679e+00	 1.6954395e-01	 1.1918355e+00	 1.6036551e-01	[ 1.0877931e+00]	 2.1350765e-01


.. parsed-literal::

      56	 1.1556461e+00	 1.6610597e-01	 1.2047550e+00	 1.5799075e-01	[ 1.0930926e+00]	 2.1083379e-01


.. parsed-literal::

      57	 1.1666366e+00	 1.6542124e-01	 1.2156993e+00	 1.5790176e-01	[ 1.1009909e+00]	 2.0725346e-01


.. parsed-literal::

      58	 1.1753827e+00	 1.6372098e-01	 1.2249521e+00	 1.5681526e-01	[ 1.1065617e+00]	 2.1154928e-01


.. parsed-literal::

      59	 1.1849378e+00	 1.6190578e-01	 1.2347609e+00	 1.5635308e-01	[ 1.1142872e+00]	 2.1361065e-01


.. parsed-literal::

      60	 1.2058103e+00	 1.5573027e-01	 1.2571007e+00	 1.5427632e-01	[ 1.1324947e+00]	 2.0887351e-01
      61	 1.2103773e+00	 1.5350698e-01	 1.2616437e+00	 1.5307307e-01	[ 1.1391794e+00]	 2.0640182e-01


.. parsed-literal::

      62	 1.2243395e+00	 1.5447177e-01	 1.2748342e+00	 1.5343892e-01	[ 1.1564667e+00]	 1.9326663e-01


.. parsed-literal::

      63	 1.2310116e+00	 1.5351338e-01	 1.2817595e+00	 1.5297419e-01	[ 1.1617404e+00]	 2.0946598e-01
      64	 1.2392689e+00	 1.5275959e-01	 1.2902040e+00	 1.5279551e-01	[ 1.1720403e+00]	 1.9746733e-01


.. parsed-literal::

      65	 1.2512621e+00	 1.5043935e-01	 1.3024558e+00	 1.5141842e-01	[ 1.1829753e+00]	 1.9656515e-01
      66	 1.2554521e+00	 1.4928689e-01	 1.3070969e+00	 1.5102409e-01	  1.1789541e+00 	 1.9679832e-01


.. parsed-literal::

      67	 1.2653587e+00	 1.4955961e-01	 1.3165138e+00	 1.5043003e-01	[ 1.1915487e+00]	 2.0211673e-01


.. parsed-literal::

      68	 1.2698381e+00	 1.4916145e-01	 1.3210452e+00	 1.4982279e-01	[ 1.1939156e+00]	 2.1271086e-01


.. parsed-literal::

      69	 1.2774600e+00	 1.4911551e-01	 1.3288749e+00	 1.4909547e-01	[ 1.1971955e+00]	 2.1398020e-01


.. parsed-literal::

      70	 1.2848573e+00	 1.4770586e-01	 1.3364533e+00	 1.4787525e-01	[ 1.1974765e+00]	 2.1020961e-01


.. parsed-literal::

      71	 1.2928414e+00	 1.4769321e-01	 1.3446011e+00	 1.4778037e-01	[ 1.2044770e+00]	 2.1317625e-01


.. parsed-literal::

      72	 1.2981332e+00	 1.4729625e-01	 1.3499656e+00	 1.4762874e-01	[ 1.2093740e+00]	 2.1518683e-01
      73	 1.3064457e+00	 1.4653935e-01	 1.3585534e+00	 1.4739398e-01	[ 1.2133877e+00]	 1.7765617e-01


.. parsed-literal::

      74	 1.3171025e+00	 1.4555249e-01	 1.3693594e+00	 1.4743558e-01	[ 1.2193428e+00]	 2.0356393e-01


.. parsed-literal::

      75	 1.3237178e+00	 1.4479140e-01	 1.3762128e+00	 1.4718164e-01	[ 1.2216943e+00]	 3.2489944e-01


.. parsed-literal::

      76	 1.3309236e+00	 1.4386596e-01	 1.3834185e+00	 1.4646067e-01	[ 1.2236086e+00]	 2.1434832e-01


.. parsed-literal::

      77	 1.3366862e+00	 1.4314592e-01	 1.3892075e+00	 1.4554463e-01	[ 1.2276024e+00]	 2.1011019e-01


.. parsed-literal::

      78	 1.3455191e+00	 1.4220206e-01	 1.3982226e+00	 1.4444282e-01	[ 1.2319491e+00]	 2.0615578e-01
      79	 1.3490598e+00	 1.4221983e-01	 1.4021003e+00	 1.4279856e-01	[ 1.2383809e+00]	 2.0189834e-01


.. parsed-literal::

      80	 1.3577217e+00	 1.4174069e-01	 1.4104996e+00	 1.4292004e-01	[ 1.2473467e+00]	 2.0282650e-01


.. parsed-literal::

      81	 1.3610626e+00	 1.4174917e-01	 1.4138626e+00	 1.4296740e-01	[ 1.2497931e+00]	 2.0147371e-01


.. parsed-literal::

      82	 1.3663179e+00	 1.4161603e-01	 1.4192989e+00	 1.4229660e-01	[ 1.2522670e+00]	 2.1552277e-01


.. parsed-literal::

      83	 1.3712281e+00	 1.4269658e-01	 1.4244665e+00	 1.4236132e-01	  1.2478155e+00 	 2.1483111e-01


.. parsed-literal::

      84	 1.3774885e+00	 1.4162604e-01	 1.4308188e+00	 1.4076775e-01	  1.2496443e+00 	 2.1042085e-01
      85	 1.3812155e+00	 1.4097748e-01	 1.4344932e+00	 1.3972916e-01	  1.2519633e+00 	 1.8870449e-01


.. parsed-literal::

      86	 1.3852838e+00	 1.4050176e-01	 1.4386475e+00	 1.3903093e-01	  1.2498116e+00 	 1.7298102e-01


.. parsed-literal::

      87	 1.3906094e+00	 1.3980537e-01	 1.4440295e+00	 1.3925476e-01	  1.2474941e+00 	 2.1831989e-01
      88	 1.3959083e+00	 1.3942529e-01	 1.4493334e+00	 1.3936926e-01	  1.2507364e+00 	 1.9269681e-01


.. parsed-literal::

      89	 1.4010833e+00	 1.3884296e-01	 1.4545052e+00	 1.3948035e-01	[ 1.2532594e+00]	 2.0292592e-01
      90	 1.4058108e+00	 1.3832542e-01	 1.4592567e+00	 1.3943793e-01	[ 1.2575014e+00]	 2.0456457e-01


.. parsed-literal::

      91	 1.4111554e+00	 1.3760597e-01	 1.4645505e+00	 1.3848974e-01	[ 1.2586906e+00]	 2.0068407e-01


.. parsed-literal::

      92	 1.4159131e+00	 1.3723582e-01	 1.4692243e+00	 1.3739135e-01	[ 1.2665114e+00]	 2.1459436e-01
      93	 1.4195836e+00	 1.3694717e-01	 1.4729097e+00	 1.3634279e-01	[ 1.2676065e+00]	 2.0676565e-01


.. parsed-literal::

      94	 1.4231732e+00	 1.3654781e-01	 1.4767125e+00	 1.3472044e-01	  1.2655517e+00 	 1.8352365e-01


.. parsed-literal::

      95	 1.4272112e+00	 1.3605073e-01	 1.4808042e+00	 1.3430801e-01	  1.2623247e+00 	 2.1986437e-01


.. parsed-literal::

      96	 1.4299838e+00	 1.3569120e-01	 1.4836187e+00	 1.3442317e-01	  1.2606656e+00 	 2.0709252e-01


.. parsed-literal::

      97	 1.4330593e+00	 1.3524653e-01	 1.4867654e+00	 1.3427464e-01	  1.2595727e+00 	 2.0970440e-01


.. parsed-literal::

      98	 1.4367727e+00	 1.3453725e-01	 1.4906172e+00	 1.3413003e-01	  1.2595535e+00 	 2.0361495e-01


.. parsed-literal::

      99	 1.4406089e+00	 1.3427453e-01	 1.4944958e+00	 1.3360806e-01	  1.2616243e+00 	 2.2083545e-01


.. parsed-literal::

     100	 1.4437846e+00	 1.3400228e-01	 1.4977516e+00	 1.3296044e-01	  1.2630793e+00 	 2.0461345e-01


.. parsed-literal::

     101	 1.4474440e+00	 1.3400168e-01	 1.5014659e+00	 1.3241667e-01	  1.2641660e+00 	 2.1511459e-01


.. parsed-literal::

     102	 1.4503339e+00	 1.3369512e-01	 1.5045957e+00	 1.3233711e-01	  1.2601299e+00 	 2.0579004e-01
     103	 1.4532534e+00	 1.3359684e-01	 1.5074034e+00	 1.3230170e-01	  1.2619909e+00 	 1.9577241e-01


.. parsed-literal::

     104	 1.4552138e+00	 1.3345914e-01	 1.5093330e+00	 1.3237242e-01	  1.2607121e+00 	 2.1661162e-01
     105	 1.4579865e+00	 1.3321775e-01	 1.5120945e+00	 1.3250380e-01	  1.2624960e+00 	 1.9426394e-01


.. parsed-literal::

     106	 1.4605570e+00	 1.3281865e-01	 1.5147787e+00	 1.3239208e-01	  1.2552860e+00 	 2.1616769e-01


.. parsed-literal::

     107	 1.4638618e+00	 1.3270164e-01	 1.5180134e+00	 1.3230710e-01	  1.2646901e+00 	 2.4218798e-01


.. parsed-literal::

     108	 1.4655853e+00	 1.3262671e-01	 1.5197472e+00	 1.3208181e-01	[ 1.2683747e+00]	 2.1974969e-01


.. parsed-literal::

     109	 1.4677945e+00	 1.3258100e-01	 1.5219954e+00	 1.3179651e-01	[ 1.2708374e+00]	 2.0610189e-01


.. parsed-literal::

     110	 1.4714929e+00	 1.3252299e-01	 1.5257988e+00	 1.3122315e-01	[ 1.2723802e+00]	 2.0900226e-01


.. parsed-literal::

     111	 1.4740492e+00	 1.3264181e-01	 1.5284399e+00	 1.3110468e-01	[ 1.2753074e+00]	 3.3816195e-01


.. parsed-literal::

     112	 1.4770470e+00	 1.3260438e-01	 1.5314628e+00	 1.3079419e-01	  1.2743049e+00 	 2.1357441e-01
     113	 1.4801265e+00	 1.3263924e-01	 1.5346010e+00	 1.3045425e-01	  1.2732939e+00 	 1.9690728e-01


.. parsed-literal::

     114	 1.4823211e+00	 1.3249535e-01	 1.5369425e+00	 1.3043788e-01	  1.2689455e+00 	 2.1747017e-01


.. parsed-literal::

     115	 1.4848436e+00	 1.3247511e-01	 1.5394681e+00	 1.3015764e-01	  1.2699822e+00 	 2.0921493e-01


.. parsed-literal::

     116	 1.4873484e+00	 1.3236173e-01	 1.5420518e+00	 1.2973075e-01	  1.2697077e+00 	 2.1361685e-01


.. parsed-literal::

     117	 1.4896298e+00	 1.3223969e-01	 1.5444410e+00	 1.2934926e-01	  1.2657689e+00 	 2.0780849e-01
     118	 1.4914256e+00	 1.3220736e-01	 1.5463783e+00	 1.2876748e-01	  1.2599142e+00 	 1.9184303e-01


.. parsed-literal::

     119	 1.4934709e+00	 1.3218877e-01	 1.5483887e+00	 1.2869375e-01	  1.2577587e+00 	 2.0964646e-01
     120	 1.4949074e+00	 1.3218527e-01	 1.5497821e+00	 1.2872146e-01	  1.2570460e+00 	 1.8752456e-01


.. parsed-literal::

     121	 1.4969362e+00	 1.3207836e-01	 1.5518403e+00	 1.2869483e-01	  1.2539576e+00 	 1.8629003e-01
     122	 1.4989182e+00	 1.3201589e-01	 1.5539304e+00	 1.2893601e-01	  1.2499163e+00 	 1.6669679e-01


.. parsed-literal::

     123	 1.5009842e+00	 1.3181751e-01	 1.5559435e+00	 1.2887338e-01	  1.2514989e+00 	 2.0874619e-01
     124	 1.5022389e+00	 1.3166805e-01	 1.5572231e+00	 1.2888581e-01	  1.2525172e+00 	 1.8441200e-01


.. parsed-literal::

     125	 1.5038806e+00	 1.3149833e-01	 1.5588993e+00	 1.2884937e-01	  1.2534927e+00 	 2.1269155e-01


.. parsed-literal::

     126	 1.5061709e+00	 1.3125423e-01	 1.5612733e+00	 1.2883557e-01	  1.2574480e+00 	 2.0659041e-01


.. parsed-literal::

     127	 1.5083101e+00	 1.3109575e-01	 1.5634809e+00	 1.2844393e-01	  1.2576737e+00 	 2.0487690e-01
     128	 1.5098244e+00	 1.3105873e-01	 1.5649459e+00	 1.2845588e-01	  1.2586500e+00 	 1.7841434e-01


.. parsed-literal::

     129	 1.5119958e+00	 1.3104224e-01	 1.5670904e+00	 1.2842269e-01	  1.2605348e+00 	 1.7938995e-01
     130	 1.5132624e+00	 1.3101596e-01	 1.5683655e+00	 1.2848673e-01	  1.2575573e+00 	 1.7473817e-01


.. parsed-literal::

     131	 1.5148934e+00	 1.3097868e-01	 1.5699822e+00	 1.2842537e-01	  1.2597695e+00 	 1.8494034e-01


.. parsed-literal::

     132	 1.5167631e+00	 1.3096949e-01	 1.5718930e+00	 1.2845706e-01	  1.2613252e+00 	 2.1113825e-01
     133	 1.5181065e+00	 1.3097757e-01	 1.5732813e+00	 1.2828736e-01	  1.2599982e+00 	 1.9411635e-01


.. parsed-literal::

     134	 1.5198073e+00	 1.3124822e-01	 1.5751107e+00	 1.2798399e-01	  1.2545037e+00 	 2.0760250e-01
     135	 1.5213123e+00	 1.3122026e-01	 1.5766771e+00	 1.2759934e-01	  1.2521802e+00 	 1.9993806e-01


.. parsed-literal::

     136	 1.5223204e+00	 1.3126003e-01	 1.5776823e+00	 1.2745436e-01	  1.2513833e+00 	 2.2008824e-01


.. parsed-literal::

     137	 1.5245570e+00	 1.3137850e-01	 1.5799928e+00	 1.2724612e-01	  1.2479509e+00 	 2.1146631e-01


.. parsed-literal::

     138	 1.5258750e+00	 1.3172122e-01	 1.5814642e+00	 1.2693838e-01	  1.2465290e+00 	 2.1968961e-01
     139	 1.5278716e+00	 1.3173720e-01	 1.5833981e+00	 1.2694660e-01	  1.2469673e+00 	 1.9108057e-01


.. parsed-literal::

     140	 1.5290960e+00	 1.3173883e-01	 1.5845968e+00	 1.2692844e-01	  1.2494499e+00 	 2.0767260e-01


.. parsed-literal::

     141	 1.5305218e+00	 1.3179076e-01	 1.5860146e+00	 1.2675985e-01	  1.2526900e+00 	 2.0894337e-01
     142	 1.5323225e+00	 1.3191455e-01	 1.5878510e+00	 1.2670974e-01	  1.2550682e+00 	 1.7479658e-01


.. parsed-literal::

     143	 1.5341059e+00	 1.3209166e-01	 1.5896744e+00	 1.2631134e-01	  1.2549562e+00 	 1.9967484e-01


.. parsed-literal::

     144	 1.5352481e+00	 1.3202828e-01	 1.5908305e+00	 1.2627828e-01	  1.2544721e+00 	 2.1226215e-01


.. parsed-literal::

     145	 1.5364224e+00	 1.3202264e-01	 1.5921141e+00	 1.2638345e-01	  1.2493032e+00 	 2.0689750e-01


.. parsed-literal::

     146	 1.5374930e+00	 1.3191866e-01	 1.5932231e+00	 1.2634266e-01	  1.2481967e+00 	 2.1480250e-01
     147	 1.5387116e+00	 1.3180647e-01	 1.5944569e+00	 1.2637679e-01	  1.2473674e+00 	 1.9900656e-01


.. parsed-literal::

     148	 1.5407117e+00	 1.3159549e-01	 1.5964795e+00	 1.2644580e-01	  1.2452700e+00 	 2.0017886e-01
     149	 1.5421437e+00	 1.3149537e-01	 1.5979279e+00	 1.2644288e-01	  1.2419982e+00 	 1.7188501e-01


.. parsed-literal::

     150	 1.5437674e+00	 1.3127630e-01	 1.5995911e+00	 1.2638609e-01	  1.2414584e+00 	 2.1683168e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.28 s, total: 2min 4s
    Wall time: 31.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f6d2d9e2fb0>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.09 s, sys: 48.9 ms, total: 2.14 s
    Wall time: 647 ms


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

