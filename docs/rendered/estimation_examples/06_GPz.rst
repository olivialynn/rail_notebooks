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
       1	-3.5098892e-01	 3.2295923e-01	-3.4122404e-01	 3.1211292e-01	[-3.1928957e-01]	 4.7106433e-01


.. parsed-literal::

       2	-2.7716700e-01	 3.1088541e-01	-2.5207819e-01	 3.0513580e-01	[-2.2480428e-01]	 2.3265553e-01


.. parsed-literal::

       3	-2.3174209e-01	 2.8992841e-01	-1.8851533e-01	 2.8642280e-01	[-1.6367867e-01]	 2.8992248e-01


.. parsed-literal::

       4	-1.9102673e-01	 2.7180341e-01	-1.4121629e-01	 2.6929524e-01	[-1.2420895e-01]	 3.0315924e-01


.. parsed-literal::

       5	-1.3929305e-01	 2.5898626e-01	-1.0970680e-01	 2.5777952e-01	[-8.7525719e-02]	 2.1478128e-01


.. parsed-literal::

       6	-7.9798784e-02	 2.5499328e-01	-5.2247728e-02	 2.4949364e-01	[-2.9874066e-02]	 2.0911622e-01
       7	-6.0833563e-02	 2.5092851e-01	-3.7547336e-02	 2.4530066e-01	[-1.3465828e-02]	 1.8477535e-01


.. parsed-literal::

       8	-4.9655289e-02	 2.4908604e-01	-2.9280636e-02	 2.4229560e-01	[-8.1451174e-04]	 2.1262789e-01


.. parsed-literal::

       9	-3.3465245e-02	 2.4600903e-01	-1.6192377e-02	 2.3787314e-01	[ 1.6458230e-02]	 2.1116948e-01


.. parsed-literal::

      10	-2.2882502e-02	 2.4385812e-01	-7.5277367e-03	 2.3614114e-01	[ 2.5899096e-02]	 2.0601511e-01
      11	-1.7515345e-02	 2.4293711e-01	-2.9179340e-03	 2.3604232e-01	[ 2.6510349e-02]	 1.8848681e-01


.. parsed-literal::

      12	-1.3203928e-02	 2.4204180e-01	 1.0773095e-03	 2.3604063e-01	  2.6424000e-02 	 2.0122862e-01
      13	-8.4390346e-03	 2.4103093e-01	 5.8802404e-03	 2.3570203e-01	[ 2.8551496e-02]	 1.8102360e-01


.. parsed-literal::

      14	 7.1474982e-02	 2.2827054e-01	 9.0933389e-02	 2.3177819e-01	[ 9.9254020e-02]	 3.0986977e-01


.. parsed-literal::

      15	 9.4729630e-02	 2.2511162e-01	 1.1759253e-01	 2.2034332e-01	[ 1.3109824e-01]	 2.1084595e-01


.. parsed-literal::

      16	 1.7632505e-01	 2.1771906e-01	 2.0207645e-01	 2.1892169e-01	[ 2.1600790e-01]	 2.0729423e-01


.. parsed-literal::

      17	 2.8031632e-01	 2.1143150e-01	 3.1168482e-01	 2.0894598e-01	[ 3.3112327e-01]	 2.1941185e-01


.. parsed-literal::

      18	 3.2725807e-01	 2.0779724e-01	 3.5916203e-01	 2.1240735e-01	[ 3.6803866e-01]	 2.1213222e-01


.. parsed-literal::

      19	 3.8776962e-01	 2.0504652e-01	 4.2075858e-01	 2.0604693e-01	[ 4.3256413e-01]	 2.2114778e-01


.. parsed-literal::

      20	 4.6406281e-01	 2.0229599e-01	 4.9802431e-01	 2.0451854e-01	[ 5.0645995e-01]	 2.1244621e-01
      21	 5.2959657e-01	 2.0537673e-01	 5.6540451e-01	 2.0394954e-01	[ 5.8577611e-01]	 1.8149710e-01


.. parsed-literal::

      22	 5.9856307e-01	 2.0017857e-01	 6.3715725e-01	 2.0051111e-01	[ 6.4838016e-01]	 1.9572520e-01


.. parsed-literal::

      23	 6.4108950e-01	 1.9510123e-01	 6.8003016e-01	 1.9675785e-01	[ 6.8894108e-01]	 2.1049500e-01


.. parsed-literal::

      24	 6.7594427e-01	 1.9187360e-01	 7.1409674e-01	 1.9374363e-01	[ 7.2423865e-01]	 2.0422888e-01


.. parsed-literal::

      25	 7.0819661e-01	 1.9031913e-01	 7.4621740e-01	 1.9000886e-01	[ 7.6090908e-01]	 2.0452452e-01


.. parsed-literal::

      26	 7.3492042e-01	 1.9114093e-01	 7.7301174e-01	 1.8754378e-01	[ 7.9043132e-01]	 2.1944666e-01


.. parsed-literal::

      27	 7.6360294e-01	 1.9106237e-01	 8.0178097e-01	 1.8602128e-01	[ 8.2165774e-01]	 2.0627928e-01
      28	 7.8665723e-01	 1.9408631e-01	 8.2603109e-01	 1.9254530e-01	[ 8.4113827e-01]	 1.8737745e-01


.. parsed-literal::

      29	 8.1061606e-01	 1.9066338e-01	 8.5133231e-01	 1.8836461e-01	[ 8.6318001e-01]	 2.0742488e-01


.. parsed-literal::

      30	 8.3001309e-01	 1.9178532e-01	 8.6995650e-01	 1.9194491e-01	[ 8.8046604e-01]	 2.0725250e-01


.. parsed-literal::

      31	 8.5579781e-01	 1.9045935e-01	 8.9585188e-01	 1.8844939e-01	[ 8.9749035e-01]	 2.1575046e-01


.. parsed-literal::

      32	 8.7295612e-01	 1.8872941e-01	 9.1305981e-01	 1.8524943e-01	[ 9.1737214e-01]	 2.1069360e-01


.. parsed-literal::

      33	 8.9032509e-01	 1.8771844e-01	 9.3095922e-01	 1.8451332e-01	[ 9.3677672e-01]	 2.1342635e-01


.. parsed-literal::

      34	 9.1805523e-01	 1.8526820e-01	 9.6095914e-01	 1.7845268e-01	[ 9.6130855e-01]	 2.1148658e-01


.. parsed-literal::

      35	 9.3483158e-01	 1.8387227e-01	 9.7793082e-01	 1.7935197e-01	[ 9.8739424e-01]	 2.0963955e-01


.. parsed-literal::

      36	 9.4762368e-01	 1.8237413e-01	 9.9071367e-01	 1.7780346e-01	[ 9.9729134e-01]	 2.1690345e-01


.. parsed-literal::

      37	 9.6379869e-01	 1.7883918e-01	 1.0078171e+00	 1.7315772e-01	[ 1.0095824e+00]	 2.1008492e-01


.. parsed-literal::

      38	 9.7719221e-01	 1.7678103e-01	 1.0217124e+00	 1.7126385e-01	[ 1.0239696e+00]	 2.0425534e-01
      39	 1.0013345e+00	 1.7171692e-01	 1.0472191e+00	 1.6710496e-01	[ 1.0533581e+00]	 1.9909954e-01


.. parsed-literal::

      40	 1.0157269e+00	 1.6998562e-01	 1.0621773e+00	 1.6780172e-01	[ 1.0621915e+00]	 2.1044254e-01


.. parsed-literal::

      41	 1.0283945e+00	 1.6929972e-01	 1.0745868e+00	 1.6774645e-01	[ 1.0740636e+00]	 2.0658588e-01
      42	 1.0446599e+00	 1.6745742e-01	 1.0912789e+00	 1.6564294e-01	[ 1.0867007e+00]	 1.9752932e-01


.. parsed-literal::

      43	 1.0536809e+00	 1.6715009e-01	 1.1000777e+00	 1.6747436e-01	[ 1.0916585e+00]	 2.0458293e-01


.. parsed-literal::

      44	 1.0623023e+00	 1.6644804e-01	 1.1085905e+00	 1.6707785e-01	[ 1.0962409e+00]	 2.0482111e-01


.. parsed-literal::

      45	 1.0766488e+00	 1.6235604e-01	 1.1236706e+00	 1.6528871e-01	[ 1.1002410e+00]	 2.1396279e-01
      46	 1.0874083e+00	 1.6077092e-01	 1.1345128e+00	 1.6409859e-01	[ 1.1036942e+00]	 1.9748211e-01


.. parsed-literal::

      47	 1.0993113e+00	 1.5853556e-01	 1.1464154e+00	 1.6211556e-01	[ 1.1157227e+00]	 1.7474031e-01


.. parsed-literal::

      48	 1.1092668e+00	 1.5606374e-01	 1.1569152e+00	 1.5943499e-01	[ 1.1239783e+00]	 2.1620917e-01
      49	 1.1185698e+00	 1.5416836e-01	 1.1665415e+00	 1.5859081e-01	[ 1.1292897e+00]	 1.7353320e-01


.. parsed-literal::

      50	 1.1306886e+00	 1.5095909e-01	 1.1795799e+00	 1.5812550e-01	[ 1.1336653e+00]	 2.1457982e-01


.. parsed-literal::

      51	 1.1405012e+00	 1.4980076e-01	 1.1893137e+00	 1.5633952e-01	[ 1.1516354e+00]	 2.1452904e-01


.. parsed-literal::

      52	 1.1469568e+00	 1.5017132e-01	 1.1952751e+00	 1.5497347e-01	[ 1.1605047e+00]	 2.0826483e-01


.. parsed-literal::

      53	 1.1561710e+00	 1.4924130e-01	 1.2044036e+00	 1.5387916e-01	[ 1.1671717e+00]	 2.1672487e-01


.. parsed-literal::

      54	 1.1670871e+00	 1.4772809e-01	 1.2157605e+00	 1.5292134e-01	[ 1.1764284e+00]	 2.1690965e-01


.. parsed-literal::

      55	 1.1760665e+00	 1.4672755e-01	 1.2251292e+00	 1.5040487e-01	[ 1.1880052e+00]	 2.1112227e-01
      56	 1.1842855e+00	 1.4574657e-01	 1.2334684e+00	 1.5033280e-01	[ 1.1969386e+00]	 1.7678285e-01


.. parsed-literal::

      57	 1.1919221e+00	 1.4457380e-01	 1.2412891e+00	 1.4885341e-01	[ 1.2061637e+00]	 2.1108508e-01


.. parsed-literal::

      58	 1.2000112e+00	 1.4373239e-01	 1.2493987e+00	 1.4649409e-01	[ 1.2174243e+00]	 2.1497250e-01
      59	 1.2073618e+00	 1.4266313e-01	 1.2568229e+00	 1.4304872e-01	[ 1.2189079e+00]	 1.8417740e-01


.. parsed-literal::

      60	 1.2132224e+00	 1.4222974e-01	 1.2625834e+00	 1.4273446e-01	[ 1.2251365e+00]	 2.1166468e-01


.. parsed-literal::

      61	 1.2175275e+00	 1.4171834e-01	 1.2668928e+00	 1.4201334e-01	[ 1.2284149e+00]	 2.0992708e-01


.. parsed-literal::

      62	 1.2243298e+00	 1.4063707e-01	 1.2738010e+00	 1.4073974e-01	[ 1.2335057e+00]	 2.1438646e-01


.. parsed-literal::

      63	 1.2336523e+00	 1.3956288e-01	 1.2832303e+00	 1.3933458e-01	[ 1.2366560e+00]	 2.1889925e-01


.. parsed-literal::

      64	 1.2382204e+00	 1.3799331e-01	 1.2884187e+00	 1.3945989e-01	[ 1.2370415e+00]	 2.1949244e-01


.. parsed-literal::

      65	 1.2479244e+00	 1.3773685e-01	 1.2978098e+00	 1.3911900e-01	[ 1.2454751e+00]	 2.1332026e-01
      66	 1.2540286e+00	 1.3723599e-01	 1.3039801e+00	 1.3906967e-01	[ 1.2485871e+00]	 1.9835448e-01


.. parsed-literal::

      67	 1.2616102e+00	 1.3654674e-01	 1.3119431e+00	 1.4008751e-01	[ 1.2489921e+00]	 2.1795011e-01


.. parsed-literal::

      68	 1.2656884e+00	 1.3578503e-01	 1.3165517e+00	 1.4122765e-01	  1.2488146e+00 	 2.1536779e-01


.. parsed-literal::

      69	 1.2717650e+00	 1.3559310e-01	 1.3224343e+00	 1.4072860e-01	[ 1.2525037e+00]	 2.1643353e-01


.. parsed-literal::

      70	 1.2766165e+00	 1.3539509e-01	 1.3273493e+00	 1.3979250e-01	[ 1.2555081e+00]	 2.1505022e-01


.. parsed-literal::

      71	 1.2812290e+00	 1.3520982e-01	 1.3320365e+00	 1.3822670e-01	[ 1.2609725e+00]	 2.0325446e-01


.. parsed-literal::

      72	 1.2888370e+00	 1.3444905e-01	 1.3400063e+00	 1.3609699e-01	  1.2589210e+00 	 2.1403217e-01


.. parsed-literal::

      73	 1.2945320e+00	 1.3421141e-01	 1.3459442e+00	 1.3443084e-01	[ 1.2723805e+00]	 2.0612073e-01


.. parsed-literal::

      74	 1.3000934e+00	 1.3368488e-01	 1.3512728e+00	 1.3440766e-01	[ 1.2734165e+00]	 2.1511483e-01
      75	 1.3051475e+00	 1.3306334e-01	 1.3562754e+00	 1.3425860e-01	[ 1.2747657e+00]	 1.8617702e-01


.. parsed-literal::

      76	 1.3106015e+00	 1.3244952e-01	 1.3618240e+00	 1.3337554e-01	[ 1.2761049e+00]	 2.0280051e-01


.. parsed-literal::

      77	 1.3148290e+00	 1.3186346e-01	 1.3661608e+00	 1.3240671e-01	[ 1.2850838e+00]	 3.1984305e-01


.. parsed-literal::

      78	 1.3196532e+00	 1.3146559e-01	 1.3712270e+00	 1.3156403e-01	  1.2831061e+00 	 2.0844913e-01


.. parsed-literal::

      79	 1.3238098e+00	 1.3124238e-01	 1.3754071e+00	 1.3098447e-01	[ 1.2862590e+00]	 2.0456457e-01


.. parsed-literal::

      80	 1.3301225e+00	 1.3093925e-01	 1.3817416e+00	 1.3047830e-01	  1.2851267e+00 	 2.0649552e-01


.. parsed-literal::

      81	 1.3355580e+00	 1.3047871e-01	 1.3872508e+00	 1.2990798e-01	[ 1.2892714e+00]	 2.0169759e-01


.. parsed-literal::

      82	 1.3403121e+00	 1.3015678e-01	 1.3919602e+00	 1.3003225e-01	[ 1.2909016e+00]	 2.0557928e-01


.. parsed-literal::

      83	 1.3469760e+00	 1.2975270e-01	 1.3986832e+00	 1.3087115e-01	  1.2882499e+00 	 2.1190286e-01


.. parsed-literal::

      84	 1.3496051e+00	 1.2964522e-01	 1.4015293e+00	 1.3144940e-01	  1.2902155e+00 	 2.1847916e-01


.. parsed-literal::

      85	 1.3527354e+00	 1.2954426e-01	 1.4045294e+00	 1.3111223e-01	[ 1.2954290e+00]	 3.0095172e-01


.. parsed-literal::

      86	 1.3551663e+00	 1.2948115e-01	 1.4070087e+00	 1.3082707e-01	[ 1.2983280e+00]	 2.1272826e-01
      87	 1.3580945e+00	 1.2936339e-01	 1.4100446e+00	 1.3054163e-01	[ 1.3016343e+00]	 2.0110941e-01


.. parsed-literal::

      88	 1.3631119e+00	 1.2919459e-01	 1.4152224e+00	 1.2992663e-01	[ 1.3037006e+00]	 2.0480943e-01
      89	 1.3650461e+00	 1.2925636e-01	 1.4173103e+00	 1.2951329e-01	[ 1.3076082e+00]	 1.7745256e-01


.. parsed-literal::

      90	 1.3717313e+00	 1.2916810e-01	 1.4237827e+00	 1.2926204e-01	[ 1.3146958e+00]	 2.0762467e-01


.. parsed-literal::

      91	 1.3739190e+00	 1.2906653e-01	 1.4258147e+00	 1.2910067e-01	  1.3135057e+00 	 2.1529865e-01
      92	 1.3766906e+00	 1.2897040e-01	 1.4284710e+00	 1.2902794e-01	  1.3140439e+00 	 1.8225670e-01


.. parsed-literal::

      93	 1.3802508e+00	 1.2934289e-01	 1.4321548e+00	 1.2926974e-01	  1.3070130e+00 	 2.1510148e-01
      94	 1.3837851e+00	 1.2927800e-01	 1.4356984e+00	 1.2924206e-01	  1.3122221e+00 	 2.0106268e-01


.. parsed-literal::

      95	 1.3863886e+00	 1.2934477e-01	 1.4384068e+00	 1.2936788e-01	[ 1.3156808e+00]	 2.0814323e-01


.. parsed-literal::

      96	 1.3897611e+00	 1.2953108e-01	 1.4419310e+00	 1.2975063e-01	[ 1.3186118e+00]	 2.1981835e-01


.. parsed-literal::

      97	 1.3938403e+00	 1.2987735e-01	 1.4460957e+00	 1.3043575e-01	[ 1.3236158e+00]	 2.0453024e-01


.. parsed-literal::

      98	 1.3970419e+00	 1.3029258e-01	 1.4492753e+00	 1.3159547e-01	[ 1.3238125e+00]	 3.2095003e-01


.. parsed-literal::

      99	 1.4003421e+00	 1.3055911e-01	 1.4524238e+00	 1.3218045e-01	[ 1.3302011e+00]	 2.1321201e-01


.. parsed-literal::

     100	 1.4029406e+00	 1.3053616e-01	 1.4548453e+00	 1.3251464e-01	[ 1.3353704e+00]	 2.0750690e-01


.. parsed-literal::

     101	 1.4057741e+00	 1.3038897e-01	 1.4576325e+00	 1.3284696e-01	[ 1.3396419e+00]	 2.1247721e-01


.. parsed-literal::

     102	 1.4086603e+00	 1.3052603e-01	 1.4606481e+00	 1.3382293e-01	[ 1.3445693e+00]	 2.0336556e-01


.. parsed-literal::

     103	 1.4114719e+00	 1.3012040e-01	 1.4635976e+00	 1.3375505e-01	[ 1.3451877e+00]	 2.0974970e-01


.. parsed-literal::

     104	 1.4133381e+00	 1.2995004e-01	 1.4655769e+00	 1.3365449e-01	[ 1.3459889e+00]	 2.1588254e-01
     105	 1.4157851e+00	 1.2992896e-01	 1.4682765e+00	 1.3383858e-01	  1.3447695e+00 	 1.8478465e-01


.. parsed-literal::

     106	 1.4185332e+00	 1.2959535e-01	 1.4711274e+00	 1.3374923e-01	[ 1.3463719e+00]	 1.9738030e-01
     107	 1.4213181e+00	 1.2969087e-01	 1.4738892e+00	 1.3405930e-01	  1.3453260e+00 	 1.7300653e-01


.. parsed-literal::

     108	 1.4233317e+00	 1.2975764e-01	 1.4757828e+00	 1.3413519e-01	  1.3440131e+00 	 2.1256495e-01
     109	 1.4251757e+00	 1.2976870e-01	 1.4775397e+00	 1.3423412e-01	  1.3442167e+00 	 1.8136287e-01


.. parsed-literal::

     110	 1.4282441e+00	 1.2973567e-01	 1.4805729e+00	 1.3404011e-01	  1.3420631e+00 	 2.1194077e-01
     111	 1.4314564e+00	 1.2971814e-01	 1.4840383e+00	 1.3404739e-01	  1.3382536e+00 	 1.8698096e-01


.. parsed-literal::

     112	 1.4340013e+00	 1.2978221e-01	 1.4866029e+00	 1.3418079e-01	  1.3345035e+00 	 2.1042991e-01


.. parsed-literal::

     113	 1.4359126e+00	 1.2980560e-01	 1.4885901e+00	 1.3399905e-01	  1.3350160e+00 	 2.1620321e-01


.. parsed-literal::

     114	 1.4381322e+00	 1.2986132e-01	 1.4909846e+00	 1.3402310e-01	  1.3334425e+00 	 2.1569324e-01
     115	 1.4403989e+00	 1.2994747e-01	 1.4933028e+00	 1.3420779e-01	  1.3334916e+00 	 1.8010855e-01


.. parsed-literal::

     116	 1.4431348e+00	 1.3002725e-01	 1.4959601e+00	 1.3457753e-01	  1.3327389e+00 	 2.0424485e-01


.. parsed-literal::

     117	 1.4464781e+00	 1.3009040e-01	 1.4991338e+00	 1.3516810e-01	  1.3305772e+00 	 2.0895433e-01


.. parsed-literal::

     118	 1.4479246e+00	 1.2996548e-01	 1.5005897e+00	 1.3537516e-01	  1.3271426e+00 	 3.2916427e-01


.. parsed-literal::

     119	 1.4493685e+00	 1.2973036e-01	 1.5020010e+00	 1.3520212e-01	  1.3284454e+00 	 2.1005845e-01


.. parsed-literal::

     120	 1.4524029e+00	 1.2913699e-01	 1.5051148e+00	 1.3463927e-01	  1.3284303e+00 	 2.1663499e-01
     121	 1.4544344e+00	 1.2877703e-01	 1.5072737e+00	 1.3457950e-01	  1.3250244e+00 	 1.9192743e-01


.. parsed-literal::

     122	 1.4568801e+00	 1.2827596e-01	 1.5098037e+00	 1.3462029e-01	  1.3230921e+00 	 1.7957902e-01


.. parsed-literal::

     123	 1.4594674e+00	 1.2799635e-01	 1.5124440e+00	 1.3537880e-01	  1.3183240e+00 	 2.2032356e-01


.. parsed-literal::

     124	 1.4602280e+00	 1.2766132e-01	 1.5133153e+00	 1.3722421e-01	  1.3014667e+00 	 2.1227622e-01
     125	 1.4626412e+00	 1.2770225e-01	 1.5155986e+00	 1.3644358e-01	  1.3100680e+00 	 1.8062758e-01


.. parsed-literal::

     126	 1.4637972e+00	 1.2763641e-01	 1.5166999e+00	 1.3616615e-01	  1.3110554e+00 	 2.0994163e-01


.. parsed-literal::

     127	 1.4663102e+00	 1.2729089e-01	 1.5191963e+00	 1.3528735e-01	  1.3076641e+00 	 2.1300912e-01


.. parsed-literal::

     128	 1.4678861e+00	 1.2685461e-01	 1.5209072e+00	 1.3457299e-01	  1.3022757e+00 	 2.1787620e-01


.. parsed-literal::

     129	 1.4704109e+00	 1.2660549e-01	 1.5234286e+00	 1.3425051e-01	  1.2970377e+00 	 2.0278478e-01
     130	 1.4722667e+00	 1.2632857e-01	 1.5253603e+00	 1.3387437e-01	  1.2919589e+00 	 1.7760563e-01


.. parsed-literal::

     131	 1.4736159e+00	 1.2609381e-01	 1.5268006e+00	 1.3359931e-01	  1.2868508e+00 	 1.9163609e-01


.. parsed-literal::

     132	 1.4750929e+00	 1.2563555e-01	 1.5284875e+00	 1.3288011e-01	  1.2851811e+00 	 2.1159267e-01


.. parsed-literal::

     133	 1.4775117e+00	 1.2545430e-01	 1.5308956e+00	 1.3286973e-01	  1.2794588e+00 	 2.1172404e-01
     134	 1.4788917e+00	 1.2538868e-01	 1.5322301e+00	 1.3291292e-01	  1.2782055e+00 	 1.9578505e-01


.. parsed-literal::

     135	 1.4805564e+00	 1.2523640e-01	 1.5338686e+00	 1.3291127e-01	  1.2745676e+00 	 2.1052575e-01


.. parsed-literal::

     136	 1.4834630e+00	 1.2497369e-01	 1.5367891e+00	 1.3321954e-01	  1.2622061e+00 	 2.1398544e-01


.. parsed-literal::

     137	 1.4851642e+00	 1.2473968e-01	 1.5385587e+00	 1.3304213e-01	  1.2597030e+00 	 3.1909585e-01


.. parsed-literal::

     138	 1.4871332e+00	 1.2450920e-01	 1.5405684e+00	 1.3337064e-01	  1.2505622e+00 	 2.2021508e-01
     139	 1.4884637e+00	 1.2436196e-01	 1.5419495e+00	 1.3333921e-01	  1.2502671e+00 	 1.9171619e-01


.. parsed-literal::

     140	 1.4904194e+00	 1.2393642e-01	 1.5440228e+00	 1.3309686e-01	  1.2509676e+00 	 1.9933724e-01
     141	 1.4922327e+00	 1.2338602e-01	 1.5461140e+00	 1.3250700e-01	  1.2429451e+00 	 1.9453692e-01


.. parsed-literal::

     142	 1.4942648e+00	 1.2316878e-01	 1.5480808e+00	 1.3215552e-01	  1.2498316e+00 	 2.1082592e-01
     143	 1.4957261e+00	 1.2292776e-01	 1.5495067e+00	 1.3178274e-01	  1.2476281e+00 	 1.8913913e-01


.. parsed-literal::

     144	 1.4971065e+00	 1.2274157e-01	 1.5508798e+00	 1.3121411e-01	  1.2504883e+00 	 1.6926289e-01


.. parsed-literal::

     145	 1.4988428e+00	 1.2241049e-01	 1.5526329e+00	 1.3055271e-01	  1.2397215e+00 	 2.1135163e-01


.. parsed-literal::

     146	 1.5003921e+00	 1.2228367e-01	 1.5541749e+00	 1.3016595e-01	  1.2428459e+00 	 2.1464825e-01


.. parsed-literal::

     147	 1.5021493e+00	 1.2208251e-01	 1.5559466e+00	 1.2976107e-01	  1.2469371e+00 	 2.0160770e-01


.. parsed-literal::

     148	 1.5037312e+00	 1.2177658e-01	 1.5575716e+00	 1.2953037e-01	  1.2550981e+00 	 2.1334982e-01


.. parsed-literal::

     149	 1.5055377e+00	 1.2158803e-01	 1.5593842e+00	 1.2940257e-01	  1.2601518e+00 	 2.0932651e-01


.. parsed-literal::

     150	 1.5071610e+00	 1.2124444e-01	 1.5610691e+00	 1.2948624e-01	  1.2605380e+00 	 2.1234584e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.15 s, total: 2min 8s
    Wall time: 32.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fa7e49ab790>



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
    CPU times: user 2.18 s, sys: 63.9 ms, total: 2.24 s
    Wall time: 698 ms


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

