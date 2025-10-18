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
       1	-3.4944677e-01	 3.2225998e-01	-3.3982307e-01	 3.1323424e-01	[-3.2321969e-01]	 4.5761824e-01


.. parsed-literal::

       2	-2.7910782e-01	 3.1208916e-01	-2.5552933e-01	 3.0375002e-01	[-2.3048728e-01]	 2.2948933e-01


.. parsed-literal::

       3	-2.3412907e-01	 2.9043553e-01	-1.9144258e-01	 2.8273507e-01	[-1.6045898e-01]	 2.9285860e-01
       4	-1.9444487e-01	 2.6696399e-01	-1.5233350e-01	 2.6074080e-01	[-1.1626672e-01]	 1.5973854e-01


.. parsed-literal::

       5	-1.0448724e-01	 2.5767667e-01	-7.2009093e-02	 2.5136734e-01	[-4.4125960e-02]	 1.9642186e-01


.. parsed-literal::

       6	-7.1769102e-02	 2.5236728e-01	-4.3159530e-02	 2.4700229e-01	[-2.4231670e-02]	 2.0311356e-01
       7	-5.4465961e-02	 2.4960578e-01	-3.1308145e-02	 2.4547153e-01	[-1.5290119e-02]	 1.9718146e-01


.. parsed-literal::

       8	-4.1536792e-02	 2.4736800e-01	-2.1963747e-02	 2.4396482e-01	[-8.3048361e-03]	 2.0445919e-01
       9	-2.9133139e-02	 2.4502339e-01	-1.1956934e-02	 2.4208500e-01	[ 4.2433219e-05]	 1.9167280e-01


.. parsed-literal::

      10	-1.7263476e-02	 2.4268737e-01	-1.8104140e-03	 2.4023411e-01	[ 8.3607387e-03]	 1.8552160e-01


.. parsed-literal::

      11	-1.3877111e-02	 2.4243546e-01	 2.7940974e-04	 2.3964791e-01	[ 1.1299261e-02]	 2.1004128e-01


.. parsed-literal::

      12	-1.0241019e-02	 2.4161193e-01	 3.5998753e-03	 2.3919445e-01	[ 1.3461542e-02]	 2.1484590e-01


.. parsed-literal::

      13	-7.0297255e-03	 2.4096515e-01	 6.7852144e-03	 2.3851673e-01	[ 1.7090684e-02]	 2.1436882e-01


.. parsed-literal::

      14	-1.6723885e-03	 2.3978601e-01	 1.2875439e-02	 2.3678770e-01	[ 2.5002353e-02]	 2.0379090e-01


.. parsed-literal::

      15	 4.9979321e-02	 2.3115820e-01	 6.7234198e-02	 2.2675378e-01	[ 8.3192014e-02]	 2.0244479e-01


.. parsed-literal::

      16	 7.3043457e-02	 2.2883845e-01	 9.2866767e-02	 2.2303223e-01	[ 1.0253559e-01]	 3.2182860e-01


.. parsed-literal::

      17	 1.4500358e-01	 2.2375874e-01	 1.6642899e-01	 2.1710578e-01	[ 1.7750011e-01]	 2.1669626e-01


.. parsed-literal::

      18	 2.7435447e-01	 2.1954226e-01	 3.0378195e-01	 2.1171902e-01	[ 3.1415742e-01]	 2.0824909e-01
      19	 3.2853655e-01	 2.1768282e-01	 3.5803727e-01	 2.1117258e-01	[ 3.6072546e-01]	 1.7096376e-01


.. parsed-literal::

      20	 3.6499046e-01	 2.1421756e-01	 3.9435161e-01	 2.0753277e-01	[ 3.9341079e-01]	 2.0510578e-01
      21	 4.2603591e-01	 2.1047578e-01	 4.5641455e-01	 2.0394755e-01	[ 4.5260969e-01]	 1.9914031e-01


.. parsed-literal::

      22	 5.2417509e-01	 2.0849464e-01	 5.5803190e-01	 2.0099640e-01	[ 5.4290842e-01]	 2.0066714e-01


.. parsed-literal::

      23	 5.7687674e-01	 2.1048588e-01	 6.1561993e-01	 2.0187516e-01	[ 5.7380091e-01]	 2.1523356e-01
      24	 6.1479124e-01	 2.0960996e-01	 6.5252392e-01	 2.0108481e-01	[ 6.0836955e-01]	 1.8820119e-01


.. parsed-literal::

      25	 6.4009292e-01	 2.0611234e-01	 6.7645676e-01	 1.9764865e-01	[ 6.3457848e-01]	 1.8921924e-01


.. parsed-literal::

      26	 6.6360354e-01	 2.0669933e-01	 6.9834307e-01	 2.0152448e-01	[ 6.6200972e-01]	 2.1340156e-01


.. parsed-literal::

      27	 6.9204254e-01	 2.0508921e-01	 7.2638434e-01	 1.9973407e-01	[ 6.7941556e-01]	 2.0653248e-01
      28	 7.1442483e-01	 2.0231734e-01	 7.5070550e-01	 1.9557179e-01	[ 6.9775285e-01]	 1.8477273e-01


.. parsed-literal::

      29	 7.3829120e-01	 2.0432605e-01	 7.7502316e-01	 1.9997243e-01	[ 7.2566647e-01]	 2.0297337e-01


.. parsed-literal::

      30	 7.6275174e-01	 2.0392563e-01	 7.9998017e-01	 1.9974703e-01	[ 7.5704714e-01]	 2.0311809e-01
      31	 7.7774822e-01	 2.0254309e-01	 8.1564514e-01	 1.9850176e-01	[ 7.6984350e-01]	 1.8128085e-01


.. parsed-literal::

      32	 7.9335322e-01	 2.0238183e-01	 8.3143639e-01	 1.9746069e-01	[ 7.8525701e-01]	 1.9862819e-01
      33	 8.1225827e-01	 2.0178766e-01	 8.5077569e-01	 1.9708730e-01	[ 8.0065069e-01]	 1.8295097e-01


.. parsed-literal::

      34	 8.3046935e-01	 2.0311485e-01	 8.6914721e-01	 1.9825479e-01	[ 8.2066109e-01]	 1.9904923e-01


.. parsed-literal::

      35	 8.5057797e-01	 2.0072183e-01	 8.8931946e-01	 1.9512284e-01	[ 8.4035990e-01]	 2.0979595e-01


.. parsed-literal::

      36	 8.7865512e-01	 1.9473053e-01	 9.1793919e-01	 1.8749560e-01	[ 8.7155721e-01]	 2.0505548e-01


.. parsed-literal::

      37	 8.9628471e-01	 1.9369311e-01	 9.3648343e-01	 1.8413716e-01	[ 8.8717214e-01]	 2.1957946e-01
      38	 9.1596195e-01	 1.9260709e-01	 9.5642664e-01	 1.8420977e-01	[ 9.0246887e-01]	 2.0029306e-01


.. parsed-literal::

      39	 9.2852219e-01	 1.9038471e-01	 9.6987986e-01	 1.8234343e-01	[ 9.1331762e-01]	 1.8771267e-01


.. parsed-literal::

      40	 9.4327502e-01	 1.8800290e-01	 9.8585655e-01	 1.8036134e-01	[ 9.2725901e-01]	 2.1101499e-01


.. parsed-literal::

      41	 9.5993341e-01	 1.8690720e-01	 1.0037186e+00	 1.8016593e-01	[ 9.4259903e-01]	 2.0559740e-01


.. parsed-literal::

      42	 9.7663553e-01	 1.8572601e-01	 1.0205682e+00	 1.7982277e-01	[ 9.5834454e-01]	 2.0352697e-01


.. parsed-literal::

      43	 9.9421366e-01	 1.8585886e-01	 1.0379857e+00	 1.8031365e-01	[ 9.8008980e-01]	 2.1902919e-01
      44	 1.0061761e+00	 1.8366970e-01	 1.0505320e+00	 1.7890448e-01	[ 9.8415121e-01]	 1.8907428e-01


.. parsed-literal::

      45	 1.0181599e+00	 1.8280303e-01	 1.0623544e+00	 1.7788601e-01	[ 9.9875371e-01]	 2.0179009e-01
      46	 1.0243400e+00	 1.8280278e-01	 1.0684945e+00	 1.7800673e-01	[ 1.0025522e+00]	 1.9584155e-01


.. parsed-literal::

      47	 1.0333859e+00	 1.8213339e-01	 1.0783124e+00	 1.7737714e-01	[ 1.0059505e+00]	 2.1308422e-01
      48	 1.0446715e+00	 1.8180970e-01	 1.0904595e+00	 1.7713388e-01	[ 1.0088979e+00]	 1.9143009e-01


.. parsed-literal::

      49	 1.0550740e+00	 1.8068937e-01	 1.1013254e+00	 1.7525639e-01	[ 1.0146503e+00]	 2.0562482e-01


.. parsed-literal::

      50	 1.0659616e+00	 1.7913923e-01	 1.1120621e+00	 1.7319709e-01	[ 1.0279602e+00]	 2.1141267e-01


.. parsed-literal::

      51	 1.0762723e+00	 1.7800995e-01	 1.1224141e+00	 1.7058426e-01	[ 1.0414783e+00]	 2.0445538e-01


.. parsed-literal::

      52	 1.0815210e+00	 1.7711011e-01	 1.1276765e+00	 1.6922209e-01	[ 1.0538063e+00]	 2.1402645e-01
      53	 1.0876860e+00	 1.7643092e-01	 1.1338384e+00	 1.6847826e-01	[ 1.0589496e+00]	 2.0418191e-01


.. parsed-literal::

      54	 1.0949507e+00	 1.7569167e-01	 1.1413869e+00	 1.6740198e-01	[ 1.0636365e+00]	 2.0826674e-01


.. parsed-literal::

      55	 1.1023793e+00	 1.7504840e-01	 1.1489272e+00	 1.6644243e-01	[ 1.0717918e+00]	 2.1954942e-01


.. parsed-literal::

      56	 1.1163390e+00	 1.7411418e-01	 1.1636389e+00	 1.6441756e-01	[ 1.0838024e+00]	 2.0120168e-01


.. parsed-literal::

      57	 1.1276574e+00	 1.7328499e-01	 1.1753112e+00	 1.6308781e-01	[ 1.0956940e+00]	 2.0095062e-01


.. parsed-literal::

      58	 1.1337303e+00	 1.7256279e-01	 1.1812710e+00	 1.6260967e-01	[ 1.0986631e+00]	 2.0704484e-01


.. parsed-literal::

      59	 1.1418382e+00	 1.7141568e-01	 1.1895887e+00	 1.6177983e-01	[ 1.1017873e+00]	 2.0127296e-01


.. parsed-literal::

      60	 1.1482988e+00	 1.7004982e-01	 1.1963994e+00	 1.6031681e-01	[ 1.1035260e+00]	 2.1178317e-01
      61	 1.1561581e+00	 1.6846499e-01	 1.2045356e+00	 1.5927119e-01	[ 1.1081355e+00]	 1.9614673e-01


.. parsed-literal::

      62	 1.1632875e+00	 1.6693771e-01	 1.2119471e+00	 1.5798988e-01	[ 1.1156681e+00]	 2.0089507e-01
      63	 1.1723372e+00	 1.6540375e-01	 1.2213227e+00	 1.5697746e-01	[ 1.1247351e+00]	 1.8441391e-01


.. parsed-literal::

      64	 1.1822552e+00	 1.6371100e-01	 1.2314712e+00	 1.5585492e-01	[ 1.1417883e+00]	 2.0666337e-01
      65	 1.1900490e+00	 1.6214338e-01	 1.2394940e+00	 1.5465500e-01	[ 1.1459203e+00]	 2.0333624e-01


.. parsed-literal::

      66	 1.1953616e+00	 1.6224156e-01	 1.2447895e+00	 1.5483921e-01	[ 1.1538849e+00]	 1.9539332e-01


.. parsed-literal::

      67	 1.2007114e+00	 1.6138491e-01	 1.2502929e+00	 1.5403278e-01	[ 1.1581140e+00]	 2.0462465e-01
      68	 1.2112065e+00	 1.5993398e-01	 1.2613097e+00	 1.5253773e-01	[ 1.1637100e+00]	 1.8968010e-01


.. parsed-literal::

      69	 1.2173865e+00	 1.5929406e-01	 1.2675697e+00	 1.5181077e-01	[ 1.1732882e+00]	 1.6800427e-01


.. parsed-literal::

      70	 1.2238391e+00	 1.5836106e-01	 1.2740041e+00	 1.5106243e-01	[ 1.1817746e+00]	 2.0327997e-01
      71	 1.2286517e+00	 1.5747677e-01	 1.2788655e+00	 1.5030321e-01	[ 1.1874136e+00]	 1.9691825e-01


.. parsed-literal::

      72	 1.2341037e+00	 1.5623966e-01	 1.2844927e+00	 1.4959793e-01	[ 1.1938031e+00]	 1.9536519e-01


.. parsed-literal::

      73	 1.2415554e+00	 1.5490352e-01	 1.2923559e+00	 1.4867502e-01	[ 1.1967900e+00]	 2.1902227e-01


.. parsed-literal::

      74	 1.2488492e+00	 1.5370714e-01	 1.2997798e+00	 1.4822046e-01	[ 1.1994942e+00]	 2.0646095e-01
      75	 1.2560812e+00	 1.5327467e-01	 1.3072433e+00	 1.4865398e-01	  1.1938281e+00 	 1.9732547e-01


.. parsed-literal::

      76	 1.2600297e+00	 1.5252368e-01	 1.3111668e+00	 1.4846984e-01	[ 1.1996508e+00]	 1.8804193e-01


.. parsed-literal::

      77	 1.2646468e+00	 1.5241213e-01	 1.3155435e+00	 1.4812438e-01	[ 1.2019273e+00]	 2.0650363e-01
      78	 1.2686484e+00	 1.5192856e-01	 1.3195165e+00	 1.4785046e-01	[ 1.2021660e+00]	 2.0140386e-01


.. parsed-literal::

      79	 1.2728678e+00	 1.5123299e-01	 1.3237382e+00	 1.4760487e-01	[ 1.2066567e+00]	 2.0150852e-01


.. parsed-literal::

      80	 1.2784331e+00	 1.4981689e-01	 1.3293568e+00	 1.4691484e-01	[ 1.2164244e+00]	 2.0951843e-01


.. parsed-literal::

      81	 1.2838723e+00	 1.4877389e-01	 1.3349014e+00	 1.4655783e-01	[ 1.2266919e+00]	 2.1547842e-01
      82	 1.2887671e+00	 1.4825275e-01	 1.3398169e+00	 1.4614564e-01	[ 1.2354736e+00]	 1.7891645e-01


.. parsed-literal::

      83	 1.2950641e+00	 1.4755140e-01	 1.3463229e+00	 1.4566123e-01	[ 1.2378804e+00]	 2.0664477e-01
      84	 1.3001493e+00	 1.4679996e-01	 1.3515004e+00	 1.4488916e-01	[ 1.2469750e+00]	 1.6761303e-01


.. parsed-literal::

      85	 1.3049398e+00	 1.4645013e-01	 1.3562554e+00	 1.4480653e-01	[ 1.2479710e+00]	 2.0356774e-01
      86	 1.3094984e+00	 1.4585524e-01	 1.3608408e+00	 1.4463180e-01	  1.2470495e+00 	 1.7712498e-01


.. parsed-literal::

      87	 1.3138842e+00	 1.4514943e-01	 1.3653514e+00	 1.4394923e-01	  1.2461052e+00 	 2.0919180e-01


.. parsed-literal::

      88	 1.3198134e+00	 1.4410820e-01	 1.3715316e+00	 1.4287986e-01	  1.2454323e+00 	 2.0820904e-01


.. parsed-literal::

      89	 1.3257956e+00	 1.4307564e-01	 1.3777939e+00	 1.4167302e-01	  1.2413025e+00 	 2.0346522e-01
      90	 1.3315206e+00	 1.4227258e-01	 1.3837178e+00	 1.4087821e-01	  1.2446620e+00 	 1.7236042e-01


.. parsed-literal::

      91	 1.3361580e+00	 1.4220125e-01	 1.3882988e+00	 1.4064381e-01	[ 1.2479752e+00]	 1.9703078e-01


.. parsed-literal::

      92	 1.3398102e+00	 1.4206953e-01	 1.3918748e+00	 1.4079244e-01	[ 1.2515273e+00]	 2.1369004e-01
      93	 1.3431956e+00	 1.4173991e-01	 1.3952702e+00	 1.4077368e-01	[ 1.2532045e+00]	 1.9423819e-01


.. parsed-literal::

      94	 1.3482653e+00	 1.4096998e-01	 1.4005418e+00	 1.4010879e-01	[ 1.2574221e+00]	 2.0017219e-01


.. parsed-literal::

      95	 1.3520467e+00	 1.4052182e-01	 1.4044133e+00	 1.3979477e-01	[ 1.2611052e+00]	 2.0468473e-01


.. parsed-literal::

      96	 1.3551163e+00	 1.4022234e-01	 1.4074584e+00	 1.3932043e-01	[ 1.2670720e+00]	 2.0627499e-01
      97	 1.3588447e+00	 1.3977468e-01	 1.4111765e+00	 1.3850565e-01	[ 1.2740214e+00]	 1.7203927e-01


.. parsed-literal::

      98	 1.3616862e+00	 1.3919381e-01	 1.4140596e+00	 1.3768902e-01	[ 1.2807831e+00]	 1.8508410e-01


.. parsed-literal::

      99	 1.3649763e+00	 1.3864624e-01	 1.4174001e+00	 1.3657988e-01	  1.2807038e+00 	 2.1238971e-01


.. parsed-literal::

     100	 1.3674877e+00	 1.3828060e-01	 1.4198636e+00	 1.3603994e-01	  1.2807379e+00 	 2.0370817e-01


.. parsed-literal::

     101	 1.3696193e+00	 1.3804469e-01	 1.4219848e+00	 1.3582697e-01	[ 1.2821744e+00]	 2.0840645e-01


.. parsed-literal::

     102	 1.3727458e+00	 1.3762390e-01	 1.4251474e+00	 1.3533085e-01	  1.2816873e+00 	 2.0368671e-01


.. parsed-literal::

     103	 1.3763774e+00	 1.3730264e-01	 1.4289014e+00	 1.3492734e-01	[ 1.2837278e+00]	 2.0977616e-01
     104	 1.3793122e+00	 1.3702422e-01	 1.4319130e+00	 1.3458167e-01	[ 1.2843636e+00]	 1.9698715e-01


.. parsed-literal::

     105	 1.3816619e+00	 1.3692606e-01	 1.4343762e+00	 1.3451945e-01	[ 1.2855049e+00]	 2.0136499e-01
     106	 1.3836044e+00	 1.3688692e-01	 1.4363833e+00	 1.3460773e-01	[ 1.2912631e+00]	 2.0008087e-01


.. parsed-literal::

     107	 1.3861824e+00	 1.3681310e-01	 1.4390046e+00	 1.3465002e-01	[ 1.2951348e+00]	 2.0283437e-01
     108	 1.3905096e+00	 1.3666888e-01	 1.4435307e+00	 1.3483239e-01	[ 1.2994306e+00]	 1.7764807e-01


.. parsed-literal::

     109	 1.3926438e+00	 1.3623733e-01	 1.4458619e+00	 1.3422095e-01	[ 1.3076325e+00]	 2.0456934e-01
     110	 1.3948883e+00	 1.3631693e-01	 1.4479560e+00	 1.3424731e-01	[ 1.3078824e+00]	 1.7165399e-01


.. parsed-literal::

     111	 1.3967174e+00	 1.3617963e-01	 1.4498241e+00	 1.3403498e-01	[ 1.3083990e+00]	 1.7510509e-01
     112	 1.3994430e+00	 1.3597359e-01	 1.4526621e+00	 1.3371989e-01	[ 1.3110934e+00]	 2.0238876e-01


.. parsed-literal::

     113	 1.4023046e+00	 1.3565564e-01	 1.4557215e+00	 1.3344322e-01	[ 1.3122933e+00]	 1.7289066e-01


.. parsed-literal::

     114	 1.4054821e+00	 1.3559888e-01	 1.4589361e+00	 1.3338197e-01	[ 1.3155132e+00]	 2.0704293e-01
     115	 1.4075678e+00	 1.3560315e-01	 1.4610010e+00	 1.3355832e-01	  1.3143344e+00 	 1.9798088e-01


.. parsed-literal::

     116	 1.4094932e+00	 1.3565415e-01	 1.4628947e+00	 1.3383823e-01	  1.3130701e+00 	 2.0907474e-01
     117	 1.4122714e+00	 1.3548016e-01	 1.4657992e+00	 1.3404952e-01	  1.3104425e+00 	 1.7262340e-01


.. parsed-literal::

     118	 1.4151627e+00	 1.3561014e-01	 1.4686961e+00	 1.3435341e-01	  1.3080519e+00 	 2.0383024e-01


.. parsed-literal::

     119	 1.4178226e+00	 1.3540659e-01	 1.4714235e+00	 1.3430224e-01	  1.3092952e+00 	 2.0840478e-01


.. parsed-literal::

     120	 1.4199985e+00	 1.3524801e-01	 1.4736920e+00	 1.3423680e-01	  1.3086397e+00 	 2.0387053e-01


.. parsed-literal::

     121	 1.4220066e+00	 1.3505979e-01	 1.4756958e+00	 1.3406638e-01	  1.3098604e+00 	 2.0208740e-01


.. parsed-literal::

     122	 1.4239864e+00	 1.3485245e-01	 1.4776230e+00	 1.3402017e-01	  1.3124198e+00 	 2.0982170e-01
     123	 1.4263806e+00	 1.3457765e-01	 1.4799673e+00	 1.3394659e-01	  1.3154505e+00 	 1.7669582e-01


.. parsed-literal::

     124	 1.4282758e+00	 1.3453708e-01	 1.4818847e+00	 1.3421630e-01	  1.3135376e+00 	 1.8007708e-01
     125	 1.4303102e+00	 1.3430816e-01	 1.4839279e+00	 1.3406081e-01	[ 1.3163989e+00]	 1.8443584e-01


.. parsed-literal::

     126	 1.4319827e+00	 1.3417454e-01	 1.4856702e+00	 1.3391313e-01	[ 1.3175953e+00]	 1.7106819e-01


.. parsed-literal::

     127	 1.4342975e+00	 1.3395450e-01	 1.4881462e+00	 1.3367247e-01	[ 1.3195155e+00]	 2.0917392e-01
     128	 1.4355902e+00	 1.3353204e-01	 1.4896859e+00	 1.3313828e-01	  1.3194191e+00 	 1.9788051e-01


.. parsed-literal::

     129	 1.4382812e+00	 1.3337137e-01	 1.4923474e+00	 1.3309277e-01	[ 1.3235087e+00]	 1.9853735e-01
     130	 1.4393547e+00	 1.3325777e-01	 1.4933950e+00	 1.3301986e-01	[ 1.3260670e+00]	 1.6751456e-01


.. parsed-literal::

     131	 1.4404839e+00	 1.3308940e-01	 1.4945327e+00	 1.3291866e-01	[ 1.3285990e+00]	 2.0595455e-01
     132	 1.4426416e+00	 1.3283115e-01	 1.4966948e+00	 1.3281477e-01	[ 1.3305516e+00]	 1.9461370e-01


.. parsed-literal::

     133	 1.4444224e+00	 1.3253280e-01	 1.4986029e+00	 1.3270218e-01	[ 1.3318739e+00]	 2.0595956e-01


.. parsed-literal::

     134	 1.4468532e+00	 1.3239159e-01	 1.5009991e+00	 1.3257085e-01	  1.3310461e+00 	 2.1425796e-01
     135	 1.4480624e+00	 1.3243237e-01	 1.5022017e+00	 1.3255298e-01	  1.3289810e+00 	 1.7293382e-01


.. parsed-literal::

     136	 1.4496717e+00	 1.3242552e-01	 1.5038720e+00	 1.3249259e-01	  1.3256606e+00 	 2.0009637e-01
     137	 1.4510904e+00	 1.3242858e-01	 1.5054636e+00	 1.3256305e-01	  1.3180188e+00 	 1.8308592e-01


.. parsed-literal::

     138	 1.4530293e+00	 1.3244197e-01	 1.5073533e+00	 1.3258797e-01	  1.3188527e+00 	 1.7790747e-01
     139	 1.4545991e+00	 1.3228985e-01	 1.5089246e+00	 1.3258800e-01	  1.3190535e+00 	 1.7572117e-01


.. parsed-literal::

     140	 1.4558431e+00	 1.3226491e-01	 1.5101841e+00	 1.3271403e-01	  1.3186523e+00 	 1.7478490e-01
     141	 1.4572673e+00	 1.3200499e-01	 1.5116828e+00	 1.3267336e-01	  1.3163485e+00 	 1.9656825e-01


.. parsed-literal::

     142	 1.4588702e+00	 1.3205124e-01	 1.5132313e+00	 1.3268740e-01	  1.3175130e+00 	 1.8406224e-01
     143	 1.4599894e+00	 1.3205801e-01	 1.5143197e+00	 1.3257523e-01	  1.3192980e+00 	 1.7521739e-01


.. parsed-literal::

     144	 1.4611883e+00	 1.3200912e-01	 1.5155113e+00	 1.3235445e-01	  1.3201838e+00 	 1.9519186e-01


.. parsed-literal::

     145	 1.4628878e+00	 1.3181930e-01	 1.5172909e+00	 1.3179585e-01	  1.3209343e+00 	 2.0820165e-01


.. parsed-literal::

     146	 1.4644065e+00	 1.3173911e-01	 1.5187875e+00	 1.3160600e-01	  1.3213409e+00 	 2.0375562e-01


.. parsed-literal::

     147	 1.4654693e+00	 1.3164838e-01	 1.5198703e+00	 1.3152210e-01	  1.3206278e+00 	 2.0675874e-01
     148	 1.4668848e+00	 1.3153173e-01	 1.5213346e+00	 1.3144182e-01	  1.3200946e+00 	 1.7920494e-01


.. parsed-literal::

     149	 1.4680058e+00	 1.3139112e-01	 1.5226091e+00	 1.3146745e-01	  1.3149708e+00 	 1.9866943e-01
     150	 1.4695412e+00	 1.3130031e-01	 1.5241274e+00	 1.3140256e-01	  1.3174077e+00 	 1.7795300e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min, sys: 1.05 s, total: 2min 1s
    Wall time: 30.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7eff2051a7a0>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.73 s, sys: 34 ms, total: 1.76 s
    Wall time: 537 ms


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

