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
       1	-3.4155282e-01	 3.1985618e-01	-3.3178383e-01	 3.2326585e-01	[-3.3777134e-01]	 4.6521139e-01


.. parsed-literal::

       2	-2.6829886e-01	 3.0814027e-01	-2.4355843e-01	 3.1375971e-01	[-2.5803928e-01]	 2.3375082e-01


.. parsed-literal::

       3	-2.2714521e-01	 2.8982414e-01	-1.8711900e-01	 2.9538682e-01	[-2.0694900e-01]	 2.9487729e-01
       4	-1.8419376e-01	 2.6502497e-01	-1.4257001e-01	 2.6397562e-01	[-1.3741130e-01]	 1.7682076e-01


.. parsed-literal::

       5	-1.0860912e-01	 2.5709104e-01	-7.5799700e-02	 2.6121248e-01	[-8.5365771e-02]	 2.1552920e-01
       6	-6.8739792e-02	 2.5088661e-01	-3.7817907e-02	 2.5173352e-01	[-3.9570728e-02]	 1.9669247e-01


.. parsed-literal::

       7	-5.4328377e-02	 2.4892558e-01	-2.8917991e-02	 2.5007781e-01	[-3.1398143e-02]	 2.0089364e-01


.. parsed-literal::

       8	-3.5403913e-02	 2.4569523e-01	-1.5335427e-02	 2.4713370e-01	[-1.8157035e-02]	 2.2331905e-01


.. parsed-literal::

       9	-2.4276756e-02	 2.4372174e-01	-7.3246954e-03	 2.4392496e-01	[-6.7030166e-03]	 2.1700907e-01
      10	-1.5746784e-02	 2.4234089e-01	-6.5695339e-04	 2.4109942e-01	[ 5.6066057e-03]	 1.8666935e-01


.. parsed-literal::

      11	-1.1798762e-02	 2.4158898e-01	 2.7291898e-03	 2.4037223e-01	[ 8.6033372e-03]	 2.0868683e-01


.. parsed-literal::

      12	-8.4653653e-03	 2.4108653e-01	 5.3959905e-03	 2.3990404e-01	[ 1.1152282e-02]	 2.1773934e-01


.. parsed-literal::

      13	-4.1129820e-03	 2.4022258e-01	 9.9078174e-03	 2.3920921e-01	[ 1.5328287e-02]	 2.1534109e-01


.. parsed-literal::

      14	 6.6295094e-02	 2.2913542e-01	 8.4530085e-02	 2.2894057e-01	[ 9.2051991e-02]	 3.2935762e-01


.. parsed-literal::

      15	 8.8132628e-02	 2.2578130e-01	 1.0871306e-01	 2.2373966e-01	[ 1.1412589e-01]	 3.3362007e-01


.. parsed-literal::

      16	 1.4395790e-01	 2.2253383e-01	 1.6577574e-01	 2.1858685e-01	[ 1.7402124e-01]	 2.0992327e-01


.. parsed-literal::

      17	 2.3676791e-01	 2.2017286e-01	 2.6687954e-01	 2.1274575e-01	[ 2.9096741e-01]	 2.1891594e-01


.. parsed-literal::

      18	 2.8104868e-01	 2.1694641e-01	 3.1212938e-01	 2.1180129e-01	[ 3.2377812e-01]	 2.1996427e-01


.. parsed-literal::

      19	 3.3363628e-01	 2.1185512e-01	 3.6577192e-01	 2.0583662e-01	[ 3.8246975e-01]	 2.0393562e-01


.. parsed-literal::

      20	 3.8853908e-01	 2.0764323e-01	 4.2115907e-01	 2.0177402e-01	[ 4.4172759e-01]	 2.1285558e-01
      21	 4.7283008e-01	 2.0223636e-01	 5.0695342e-01	 1.9674315e-01	[ 5.3151421e-01]	 1.8031073e-01


.. parsed-literal::

      22	 5.5002426e-01	 1.9893585e-01	 5.9139414e-01	 1.9491828e-01	[ 6.3425552e-01]	 2.1296668e-01


.. parsed-literal::

      23	 6.1779071e-01	 1.9760849e-01	 6.5536923e-01	 1.9400997e-01	[ 6.8130565e-01]	 2.1467400e-01


.. parsed-literal::

      24	 6.5008730e-01	 1.9419900e-01	 6.8776066e-01	 1.8964544e-01	[ 7.1634768e-01]	 2.1248817e-01


.. parsed-literal::

      25	 6.6987592e-01	 2.0076682e-01	 7.0563121e-01	 1.9524655e-01	[ 7.2570693e-01]	 2.0923305e-01


.. parsed-literal::

      26	 7.3003988e-01	 1.9548483e-01	 7.6605334e-01	 1.8627919e-01	[ 7.9000849e-01]	 2.2003627e-01


.. parsed-literal::

      27	 7.3948833e-01	 1.8811780e-01	 7.7899545e-01	 1.8080296e-01	[ 8.0204028e-01]	 2.1233249e-01


.. parsed-literal::

      28	 7.8181280e-01	 1.8582547e-01	 8.2014026e-01	 1.7884225e-01	[ 8.3979155e-01]	 2.1967220e-01


.. parsed-literal::

      29	 8.0848624e-01	 1.8558140e-01	 8.4643310e-01	 1.7719423e-01	[ 8.6806972e-01]	 2.2121334e-01


.. parsed-literal::

      30	 8.3378940e-01	 1.8473465e-01	 8.7236293e-01	 1.7521142e-01	[ 8.9238710e-01]	 2.0627499e-01


.. parsed-literal::

      31	 8.6524130e-01	 1.7967063e-01	 9.0559112e-01	 1.7037513e-01	[ 9.2308975e-01]	 2.1545744e-01
      32	 8.9531250e-01	 1.7808386e-01	 9.3670490e-01	 1.6935804e-01	[ 9.4487869e-01]	 2.0719671e-01


.. parsed-literal::

      33	 9.1576798e-01	 1.7562594e-01	 9.5783005e-01	 1.6878915e-01	[ 9.5739750e-01]	 2.1295309e-01


.. parsed-literal::

      34	 9.3821983e-01	 1.7278185e-01	 9.8099049e-01	 1.6685760e-01	[ 9.7218490e-01]	 2.0638299e-01


.. parsed-literal::

      35	 9.6659027e-01	 1.6912659e-01	 1.0105915e+00	 1.6398995e-01	[ 9.9653107e-01]	 2.0879221e-01


.. parsed-literal::

      36	 9.8958292e-01	 1.6431537e-01	 1.0333510e+00	 1.5811088e-01	[ 1.0221774e+00]	 2.1320462e-01


.. parsed-literal::

      37	 1.0084354e+00	 1.6238869e-01	 1.0522978e+00	 1.5519862e-01	[ 1.0473274e+00]	 2.1582723e-01


.. parsed-literal::

      38	 1.0263932e+00	 1.6016583e-01	 1.0711883e+00	 1.5121797e-01	[ 1.0673023e+00]	 2.2560096e-01
      39	 1.0395988e+00	 1.5840280e-01	 1.0847904e+00	 1.5098416e-01	[ 1.0724757e+00]	 1.7543530e-01


.. parsed-literal::

      40	 1.0531836e+00	 1.5612152e-01	 1.0984254e+00	 1.4862458e-01	[ 1.0799116e+00]	 2.1781874e-01


.. parsed-literal::

      41	 1.0694271e+00	 1.5333739e-01	 1.1152696e+00	 1.4565795e-01	[ 1.0873437e+00]	 2.1056390e-01
      42	 1.0823078e+00	 1.5188308e-01	 1.1282589e+00	 1.4352398e-01	[ 1.0944174e+00]	 1.7547822e-01


.. parsed-literal::

      43	 1.0952970e+00	 1.5115732e-01	 1.1415746e+00	 1.4267219e-01	[ 1.1065635e+00]	 2.0630312e-01


.. parsed-literal::

      44	 1.1071892e+00	 1.5073738e-01	 1.1536684e+00	 1.4220481e-01	[ 1.1184464e+00]	 2.0810962e-01


.. parsed-literal::

      45	 1.1251626e+00	 1.5006508e-01	 1.1721401e+00	 1.3990553e-01	[ 1.1381938e+00]	 2.1206474e-01


.. parsed-literal::

      46	 1.1329696e+00	 1.4975733e-01	 1.1803671e+00	 1.4037116e-01	  1.1329791e+00 	 2.1372676e-01
      47	 1.1435117e+00	 1.4727926e-01	 1.1903672e+00	 1.3769380e-01	[ 1.1494141e+00]	 1.9884062e-01


.. parsed-literal::

      48	 1.1512897e+00	 1.4569773e-01	 1.1983390e+00	 1.3613414e-01	[ 1.1569320e+00]	 1.8229270e-01


.. parsed-literal::

      49	 1.1596209e+00	 1.4397364e-01	 1.2070518e+00	 1.3517039e-01	[ 1.1633797e+00]	 2.1019244e-01


.. parsed-literal::

      50	 1.1706294e+00	 1.4250858e-01	 1.2185218e+00	 1.3479418e-01	[ 1.1724860e+00]	 2.0842671e-01


.. parsed-literal::

      51	 1.1807247e+00	 1.4092572e-01	 1.2295411e+00	 1.3405308e-01	[ 1.1852290e+00]	 2.1291327e-01


.. parsed-literal::

      52	 1.1890574e+00	 1.4070165e-01	 1.2377070e+00	 1.3370610e-01	[ 1.1915115e+00]	 2.0916247e-01


.. parsed-literal::

      53	 1.1966247e+00	 1.4041243e-01	 1.2453952e+00	 1.3340641e-01	[ 1.1987365e+00]	 2.2321391e-01


.. parsed-literal::

      54	 1.2053091e+00	 1.3980277e-01	 1.2544838e+00	 1.3264850e-01	[ 1.2080106e+00]	 2.1179891e-01
      55	 1.2147321e+00	 1.3877723e-01	 1.2642948e+00	 1.3118096e-01	[ 1.2210807e+00]	 1.9828486e-01


.. parsed-literal::

      56	 1.2228739e+00	 1.3815877e-01	 1.2725106e+00	 1.3052597e-01	[ 1.2303642e+00]	 2.1700430e-01
      57	 1.2334119e+00	 1.3759693e-01	 1.2832468e+00	 1.3016577e-01	[ 1.2418567e+00]	 1.9621086e-01


.. parsed-literal::

      58	 1.2428103e+00	 1.3732856e-01	 1.2927252e+00	 1.3053086e-01	[ 1.2537573e+00]	 2.1033406e-01


.. parsed-literal::

      59	 1.2533415e+00	 1.3750374e-01	 1.3034093e+00	 1.3163941e-01	[ 1.2624411e+00]	 2.2665811e-01


.. parsed-literal::

      60	 1.2610662e+00	 1.3704187e-01	 1.3111788e+00	 1.3158151e-01	[ 1.2715677e+00]	 2.1045280e-01


.. parsed-literal::

      61	 1.2691546e+00	 1.3647828e-01	 1.3195312e+00	 1.3125528e-01	[ 1.2775852e+00]	 2.1131587e-01


.. parsed-literal::

      62	 1.2774622e+00	 1.3579953e-01	 1.3282495e+00	 1.3081817e-01	[ 1.2799028e+00]	 2.0841193e-01


.. parsed-literal::

      63	 1.2851759e+00	 1.3506523e-01	 1.3359923e+00	 1.3003391e-01	[ 1.2822034e+00]	 2.3302436e-01


.. parsed-literal::

      64	 1.2911076e+00	 1.3462155e-01	 1.3418579e+00	 1.2976898e-01	[ 1.2836917e+00]	 2.2998667e-01


.. parsed-literal::

      65	 1.3026751e+00	 1.3375605e-01	 1.3539728e+00	 1.2990765e-01	  1.2812563e+00 	 2.2136974e-01


.. parsed-literal::

      66	 1.3087268e+00	 1.3255445e-01	 1.3601469e+00	 1.2878556e-01	[ 1.2847635e+00]	 2.1854043e-01


.. parsed-literal::

      67	 1.3170396e+00	 1.3238605e-01	 1.3683405e+00	 1.2906134e-01	[ 1.2933662e+00]	 2.0599246e-01


.. parsed-literal::

      68	 1.3250958e+00	 1.3197809e-01	 1.3764660e+00	 1.2906492e-01	[ 1.3044547e+00]	 2.1140599e-01


.. parsed-literal::

      69	 1.3319450e+00	 1.3143135e-01	 1.3835461e+00	 1.2877173e-01	[ 1.3125059e+00]	 2.2015834e-01


.. parsed-literal::

      70	 1.3376896e+00	 1.3006576e-01	 1.3896865e+00	 1.2710776e-01	[ 1.3225231e+00]	 2.1443653e-01


.. parsed-literal::

      71	 1.3448592e+00	 1.2950408e-01	 1.3967553e+00	 1.2643322e-01	[ 1.3278682e+00]	 2.1198010e-01


.. parsed-literal::

      72	 1.3494577e+00	 1.2910819e-01	 1.4013166e+00	 1.2590802e-01	[ 1.3297826e+00]	 2.1008754e-01


.. parsed-literal::

      73	 1.3564803e+00	 1.2855567e-01	 1.4085053e+00	 1.2532125e-01	[ 1.3330600e+00]	 2.1315598e-01


.. parsed-literal::

      74	 1.3605682e+00	 1.2768896e-01	 1.4131391e+00	 1.2454611e-01	  1.3323083e+00 	 2.0924044e-01


.. parsed-literal::

      75	 1.3686543e+00	 1.2752807e-01	 1.4210657e+00	 1.2482269e-01	[ 1.3402216e+00]	 2.1356201e-01


.. parsed-literal::

      76	 1.3725909e+00	 1.2730303e-01	 1.4250424e+00	 1.2511134e-01	[ 1.3434468e+00]	 2.1367717e-01
      77	 1.3781026e+00	 1.2685789e-01	 1.4307239e+00	 1.2544519e-01	[ 1.3463512e+00]	 1.9576311e-01


.. parsed-literal::

      78	 1.3844441e+00	 1.2600627e-01	 1.4373253e+00	 1.2503170e-01	[ 1.3502243e+00]	 2.0840645e-01


.. parsed-literal::

      79	 1.3892915e+00	 1.2566714e-01	 1.4421128e+00	 1.2532564e-01	  1.3495896e+00 	 2.0652008e-01


.. parsed-literal::

      80	 1.3935724e+00	 1.2534197e-01	 1.4462995e+00	 1.2463939e-01	[ 1.3535970e+00]	 2.0244479e-01
      81	 1.3968509e+00	 1.2513427e-01	 1.4495276e+00	 1.2417869e-01	[ 1.3562747e+00]	 1.7909670e-01


.. parsed-literal::

      82	 1.4002831e+00	 1.2481276e-01	 1.4529502e+00	 1.2388037e-01	  1.3559168e+00 	 2.3043919e-01
      83	 1.4052864e+00	 1.2460301e-01	 1.4579734e+00	 1.2412745e-01	[ 1.3565727e+00]	 1.9175768e-01


.. parsed-literal::

      84	 1.4101059e+00	 1.2438277e-01	 1.4627841e+00	 1.2477913e-01	[ 1.3568521e+00]	 2.0044327e-01


.. parsed-literal::

      85	 1.4138684e+00	 1.2429320e-01	 1.4665522e+00	 1.2568699e-01	  1.3554850e+00 	 2.1234751e-01
      86	 1.4172043e+00	 1.2424225e-01	 1.4698923e+00	 1.2629081e-01	[ 1.3574267e+00]	 1.7894220e-01


.. parsed-literal::

      87	 1.4213972e+00	 1.2421735e-01	 1.4741789e+00	 1.2724761e-01	[ 1.3582664e+00]	 2.0719886e-01
      88	 1.4244922e+00	 1.2389533e-01	 1.4773432e+00	 1.2728505e-01	[ 1.3599730e+00]	 1.9446349e-01


.. parsed-literal::

      89	 1.4275339e+00	 1.2373538e-01	 1.4803456e+00	 1.2717669e-01	[ 1.3617384e+00]	 1.9479418e-01


.. parsed-literal::

      90	 1.4314203e+00	 1.2349500e-01	 1.4842357e+00	 1.2693583e-01	[ 1.3624067e+00]	 2.0449591e-01


.. parsed-literal::

      91	 1.4349479e+00	 1.2324567e-01	 1.4877826e+00	 1.2669048e-01	  1.3621420e+00 	 2.1063161e-01


.. parsed-literal::

      92	 1.4371201e+00	 1.2297672e-01	 1.4902273e+00	 1.2691540e-01	  1.3540092e+00 	 2.1187949e-01


.. parsed-literal::

      93	 1.4416981e+00	 1.2274846e-01	 1.4946191e+00	 1.2641365e-01	  1.3592678e+00 	 2.0589638e-01


.. parsed-literal::

      94	 1.4431625e+00	 1.2264323e-01	 1.4960792e+00	 1.2627354e-01	  1.3599952e+00 	 2.1354604e-01


.. parsed-literal::

      95	 1.4468051e+00	 1.2243836e-01	 1.4998594e+00	 1.2618397e-01	  1.3556829e+00 	 2.1935463e-01


.. parsed-literal::

      96	 1.4488029e+00	 1.2222123e-01	 1.5019284e+00	 1.2565963e-01	  1.3566950e+00 	 3.1987166e-01


.. parsed-literal::

      97	 1.4511648e+00	 1.2207980e-01	 1.5043537e+00	 1.2555234e-01	  1.3544798e+00 	 2.0741606e-01
      98	 1.4546651e+00	 1.2184962e-01	 1.5079383e+00	 1.2537911e-01	  1.3507125e+00 	 1.9674015e-01


.. parsed-literal::

      99	 1.4561622e+00	 1.2165047e-01	 1.5095140e+00	 1.2496699e-01	  1.3453495e+00 	 2.1228886e-01
     100	 1.4585959e+00	 1.2155579e-01	 1.5119005e+00	 1.2501071e-01	  1.3473856e+00 	 1.9984484e-01


.. parsed-literal::

     101	 1.4604063e+00	 1.2138836e-01	 1.5137172e+00	 1.2497732e-01	  1.3473539e+00 	 2.0877385e-01


.. parsed-literal::

     102	 1.4621946e+00	 1.2123157e-01	 1.5155577e+00	 1.2505452e-01	  1.3460514e+00 	 2.1309566e-01
     103	 1.4651825e+00	 1.2100926e-01	 1.5186351e+00	 1.2499906e-01	  1.3442211e+00 	 1.8270755e-01


.. parsed-literal::

     104	 1.4675694e+00	 1.2102820e-01	 1.5212079e+00	 1.2612805e-01	  1.3400255e+00 	 2.1211171e-01


.. parsed-literal::

     105	 1.4698346e+00	 1.2092380e-01	 1.5234416e+00	 1.2579428e-01	  1.3404667e+00 	 2.0313382e-01


.. parsed-literal::

     106	 1.4715169e+00	 1.2085793e-01	 1.5251184e+00	 1.2555557e-01	  1.3400787e+00 	 2.2348952e-01


.. parsed-literal::

     107	 1.4735835e+00	 1.2075283e-01	 1.5272937e+00	 1.2545228e-01	  1.3365841e+00 	 2.1878862e-01


.. parsed-literal::

     108	 1.4755294e+00	 1.2054313e-01	 1.5294116e+00	 1.2523502e-01	  1.3302883e+00 	 2.0668006e-01


.. parsed-literal::

     109	 1.4776988e+00	 1.2047722e-01	 1.5315525e+00	 1.2522162e-01	  1.3315874e+00 	 2.1396446e-01
     110	 1.4789760e+00	 1.2042786e-01	 1.5328064e+00	 1.2530358e-01	  1.3321667e+00 	 1.9945574e-01


.. parsed-literal::

     111	 1.4808489e+00	 1.2035305e-01	 1.5346471e+00	 1.2525426e-01	  1.3322760e+00 	 2.2893357e-01


.. parsed-literal::

     112	 1.4819549e+00	 1.2046026e-01	 1.5357672e+00	 1.2599455e-01	  1.3261824e+00 	 2.1528506e-01


.. parsed-literal::

     113	 1.4842928e+00	 1.2035331e-01	 1.5380134e+00	 1.2554193e-01	  1.3303793e+00 	 2.0176744e-01


.. parsed-literal::

     114	 1.4853983e+00	 1.2033911e-01	 1.5391226e+00	 1.2555641e-01	  1.3301349e+00 	 2.1187186e-01


.. parsed-literal::

     115	 1.4867494e+00	 1.2033999e-01	 1.5405007e+00	 1.2578244e-01	  1.3285326e+00 	 2.0080209e-01


.. parsed-literal::

     116	 1.4890027e+00	 1.2025408e-01	 1.5428431e+00	 1.2611583e-01	  1.3254628e+00 	 2.1003485e-01


.. parsed-literal::

     117	 1.4904052e+00	 1.2022836e-01	 1.5443219e+00	 1.2657823e-01	  1.3213694e+00 	 3.0111790e-01


.. parsed-literal::

     118	 1.4923537e+00	 1.2008131e-01	 1.5463122e+00	 1.2669937e-01	  1.3197279e+00 	 2.0866776e-01
     119	 1.4941418e+00	 1.1990416e-01	 1.5481235e+00	 1.2662132e-01	  1.3196083e+00 	 1.8014288e-01


.. parsed-literal::

     120	 1.4956681e+00	 1.1963473e-01	 1.5497025e+00	 1.2636803e-01	  1.3164575e+00 	 2.0721459e-01


.. parsed-literal::

     121	 1.4971931e+00	 1.1952286e-01	 1.5512394e+00	 1.2613551e-01	  1.3171423e+00 	 2.1488905e-01


.. parsed-literal::

     122	 1.4982916e+00	 1.1947228e-01	 1.5523135e+00	 1.2598531e-01	  1.3181074e+00 	 2.1076751e-01


.. parsed-literal::

     123	 1.4998226e+00	 1.1939961e-01	 1.5538661e+00	 1.2580909e-01	  1.3169520e+00 	 2.1246338e-01


.. parsed-literal::

     124	 1.5013316e+00	 1.1931192e-01	 1.5554610e+00	 1.2583982e-01	  1.3134662e+00 	 2.2012830e-01


.. parsed-literal::

     125	 1.5029097e+00	 1.1925322e-01	 1.5570991e+00	 1.2578098e-01	  1.3123379e+00 	 2.1573544e-01


.. parsed-literal::

     126	 1.5050268e+00	 1.1918048e-01	 1.5593068e+00	 1.2582728e-01	  1.3089324e+00 	 2.0922065e-01


.. parsed-literal::

     127	 1.5064173e+00	 1.1903151e-01	 1.5607393e+00	 1.2561739e-01	  1.3095766e+00 	 2.0609021e-01


.. parsed-literal::

     128	 1.5079699e+00	 1.1893266e-01	 1.5622649e+00	 1.2555124e-01	  1.3087238e+00 	 2.1579027e-01


.. parsed-literal::

     129	 1.5094531e+00	 1.1876148e-01	 1.5637225e+00	 1.2521636e-01	  1.3090972e+00 	 2.0343757e-01


.. parsed-literal::

     130	 1.5110461e+00	 1.1859656e-01	 1.5653146e+00	 1.2507874e-01	  1.3063706e+00 	 2.1757245e-01


.. parsed-literal::

     131	 1.5125186e+00	 1.1855979e-01	 1.5667767e+00	 1.2505830e-01	  1.3069572e+00 	 2.0813704e-01
     132	 1.5144940e+00	 1.1851537e-01	 1.5688374e+00	 1.2529426e-01	  1.3007537e+00 	 1.8189025e-01


.. parsed-literal::

     133	 1.5149561e+00	 1.1867596e-01	 1.5694911e+00	 1.2569151e-01	  1.2996829e+00 	 1.7835808e-01


.. parsed-literal::

     134	 1.5164015e+00	 1.1858882e-01	 1.5708587e+00	 1.2560737e-01	  1.2994441e+00 	 2.1843076e-01


.. parsed-literal::

     135	 1.5172815e+00	 1.1854828e-01	 1.5717579e+00	 1.2559781e-01	  1.2977263e+00 	 2.0864797e-01


.. parsed-literal::

     136	 1.5185293e+00	 1.1850691e-01	 1.5730772e+00	 1.2556250e-01	  1.2952807e+00 	 2.0323992e-01


.. parsed-literal::

     137	 1.5203185e+00	 1.1845862e-01	 1.5749411e+00	 1.2550308e-01	  1.2940855e+00 	 2.1144938e-01


.. parsed-literal::

     138	 1.5214687e+00	 1.1856912e-01	 1.5763391e+00	 1.2575460e-01	  1.2813492e+00 	 2.0592380e-01


.. parsed-literal::

     139	 1.5236641e+00	 1.1847966e-01	 1.5784530e+00	 1.2548694e-01	  1.2897084e+00 	 2.0313501e-01


.. parsed-literal::

     140	 1.5246485e+00	 1.1840985e-01	 1.5793019e+00	 1.2537605e-01	  1.2914172e+00 	 2.2576475e-01
     141	 1.5255992e+00	 1.1837846e-01	 1.5802598e+00	 1.2539062e-01	  1.2891889e+00 	 2.0168829e-01


.. parsed-literal::

     142	 1.5275336e+00	 1.1819316e-01	 1.5822730e+00	 1.2526342e-01	  1.2801186e+00 	 1.7227316e-01


.. parsed-literal::

     143	 1.5284759e+00	 1.1810742e-01	 1.5833277e+00	 1.2530301e-01	  1.2793089e+00 	 3.3716416e-01


.. parsed-literal::

     144	 1.5297984e+00	 1.1799933e-01	 1.5847065e+00	 1.2525097e-01	  1.2756196e+00 	 2.1358848e-01


.. parsed-literal::

     145	 1.5313960e+00	 1.1785369e-01	 1.5864467e+00	 1.2530379e-01	  1.2643767e+00 	 2.1307516e-01


.. parsed-literal::

     146	 1.5321299e+00	 1.1786753e-01	 1.5872537e+00	 1.2555183e-01	  1.2614531e+00 	 2.0807791e-01
     147	 1.5330714e+00	 1.1784389e-01	 1.5881377e+00	 1.2554337e-01	  1.2591677e+00 	 1.8869710e-01


.. parsed-literal::

     148	 1.5345248e+00	 1.1783737e-01	 1.5895873e+00	 1.2576577e-01	  1.2491389e+00 	 2.0672750e-01
     149	 1.5353808e+00	 1.1780901e-01	 1.5904541e+00	 1.2585215e-01	  1.2397682e+00 	 2.0378852e-01


.. parsed-literal::

     150	 1.5361472e+00	 1.1777322e-01	 1.5912654e+00	 1.2585642e-01	  1.2307871e+00 	 2.1606779e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.3 s, total: 2min 8s
    Wall time: 32.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f751c4822f0>



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
    CPU times: user 1.82 s, sys: 53 ms, total: 1.87 s
    Wall time: 596 ms


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

