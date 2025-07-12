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
       1	-3.4148775e-01	 3.1972962e-01	-3.3190014e-01	 3.2447972e-01	[-3.4088753e-01]	 4.6266913e-01


.. parsed-literal::

       2	-2.6992142e-01	 3.0896459e-01	-2.4546802e-01	 3.1267607e-01	[-2.5714739e-01]	 2.3216343e-01


.. parsed-literal::

       3	-2.2395091e-01	 2.8700120e-01	-1.7946448e-01	 2.9426532e-01	[-2.0556568e-01]	 2.9102445e-01
       4	-1.8988731e-01	 2.6396449e-01	-1.4888932e-01	 2.7055407e-01	[-1.8630066e-01]	 1.8892622e-01


.. parsed-literal::

       5	-9.7483029e-02	 2.5607265e-01	-6.3797178e-02	 2.6129765e-01	[-8.6001824e-02]	 2.1949148e-01


.. parsed-literal::

       6	-6.5553669e-02	 2.5057779e-01	-3.6039469e-02	 2.5589247e-01	[-5.5766887e-02]	 2.1824121e-01


.. parsed-literal::

       7	-4.8713001e-02	 2.4802000e-01	-2.5053592e-02	 2.5216508e-01	[-4.1961602e-02]	 2.1935678e-01


.. parsed-literal::

       8	-3.6628791e-02	 2.4603284e-01	-1.6651977e-02	 2.4986549e-01	[-3.2673345e-02]	 2.2197342e-01


.. parsed-literal::

       9	-2.2935624e-02	 2.4344299e-01	-5.6502632e-03	 2.4816190e-01	[-2.3922687e-02]	 2.1185231e-01


.. parsed-literal::

      10	-1.5166495e-02	 2.4193566e-01	-1.2132276e-04	 2.4746837e-01	[-2.2678880e-02]	 2.1109962e-01
      11	-7.6423142e-03	 2.4060082e-01	 6.9171260e-03	 2.4657521e-01	[-1.4972717e-02]	 1.9366217e-01


.. parsed-literal::

      12	-5.5008562e-03	 2.4022653e-01	 8.7715345e-03	 2.4599106e-01	[-1.2317091e-02]	 2.0934439e-01
      13	-1.8364875e-04	 2.3920724e-01	 1.3900910e-02	 2.4422932e-01	[-4.6598386e-03]	 1.7892289e-01


.. parsed-literal::

      14	 8.5487938e-02	 2.2721039e-01	 1.0609725e-01	 2.2900259e-01	[ 9.3517180e-02]	 3.2569623e-01


.. parsed-literal::

      15	 1.2173870e-01	 2.2314725e-01	 1.4970383e-01	 2.2523532e-01	[ 1.4503661e-01]	 2.0585442e-01


.. parsed-literal::

      16	 2.8665661e-01	 2.1201908e-01	 3.1784557e-01	 2.1529556e-01	[ 2.9796647e-01]	 2.1458745e-01


.. parsed-literal::

      17	 3.3330563e-01	 2.0446193e-01	 3.6922457e-01	 2.0614682e-01	[ 3.4261273e-01]	 2.1048832e-01
      18	 4.0228123e-01	 1.9974815e-01	 4.3679194e-01	 2.0229154e-01	[ 4.1760647e-01]	 1.9911933e-01


.. parsed-literal::

      19	 4.5749330e-01	 1.9431900e-01	 4.9247538e-01	 1.9897663e-01	[ 4.6958396e-01]	 2.0804262e-01


.. parsed-literal::

      20	 6.1555692e-01	 1.8561292e-01	 6.5380297e-01	 1.9729139e-01	[ 6.2353927e-01]	 2.1597433e-01


.. parsed-literal::

      21	 6.6003839e-01	 1.7896332e-01	 7.0072944e-01	 1.9010528e-01	[ 6.6031261e-01]	 3.2545090e-01


.. parsed-literal::

      22	 7.0332042e-01	 1.7383228e-01	 7.4418738e-01	 1.8604539e-01	[ 7.0013397e-01]	 2.1597123e-01


.. parsed-literal::

      23	 7.5114149e-01	 1.6903580e-01	 7.9155645e-01	 1.8098847e-01	[ 7.4846964e-01]	 2.1192145e-01


.. parsed-literal::

      24	 8.0031190e-01	 1.6561254e-01	 8.4166638e-01	 1.7941667e-01	[ 8.0762362e-01]	 2.0410228e-01


.. parsed-literal::

      25	 8.4688388e-01	 1.6422901e-01	 8.8693243e-01	 1.7791017e-01	[ 8.5825082e-01]	 2.0810127e-01


.. parsed-literal::

      26	 8.8009295e-01	 1.6387703e-01	 9.2121170e-01	 1.7768891e-01	[ 8.8480539e-01]	 2.0858240e-01
      27	 9.3731623e-01	 1.5942074e-01	 9.8026753e-01	 1.7475729e-01	[ 9.3349857e-01]	 2.0040083e-01


.. parsed-literal::

      28	 9.6666944e-01	 1.5470969e-01	 1.0093049e+00	 1.7178289e-01	[ 9.4678739e-01]	 1.9793630e-01


.. parsed-literal::

      29	 9.8229170e-01	 1.5305332e-01	 1.0249565e+00	 1.7120352e-01	[ 9.6377372e-01]	 2.0829511e-01


.. parsed-literal::

      30	 1.0056180e+00	 1.5235263e-01	 1.0484354e+00	 1.7101343e-01	[ 9.8993254e-01]	 2.0484066e-01
      31	 1.0231033e+00	 1.5092835e-01	 1.0662643e+00	 1.7013929e-01	[ 1.0053251e+00]	 1.9768834e-01


.. parsed-literal::

      32	 1.0501686e+00	 1.4875365e-01	 1.0939749e+00	 1.6913865e-01	[ 1.0268535e+00]	 2.1025133e-01
      33	 1.0708672e+00	 1.4747667e-01	 1.1156452e+00	 1.6775092e-01	[ 1.0447684e+00]	 1.9738984e-01


.. parsed-literal::

      34	 1.0860243e+00	 1.4582165e-01	 1.1309378e+00	 1.6721493e-01	[ 1.0588799e+00]	 1.9642735e-01


.. parsed-literal::

      35	 1.1002907e+00	 1.4428593e-01	 1.1450665e+00	 1.6517886e-01	[ 1.0742453e+00]	 2.0687675e-01


.. parsed-literal::

      36	 1.1234344e+00	 1.4201368e-01	 1.1687771e+00	 1.6351569e-01	[ 1.0992790e+00]	 2.1026421e-01
      37	 1.1407148e+00	 1.4060461e-01	 1.1861727e+00	 1.6052647e-01	[ 1.1257328e+00]	 1.9500422e-01


.. parsed-literal::

      38	 1.1570866e+00	 1.3933034e-01	 1.2028120e+00	 1.6078945e-01	[ 1.1465417e+00]	 2.1030927e-01


.. parsed-literal::

      39	 1.1709277e+00	 1.3819041e-01	 1.2168533e+00	 1.6190232e-01	[ 1.1575739e+00]	 2.0618534e-01
      40	 1.1810471e+00	 1.3785363e-01	 1.2270401e+00	 1.6386872e-01	[ 1.1688862e+00]	 1.7685676e-01


.. parsed-literal::

      41	 1.1894261e+00	 1.3706890e-01	 1.2355064e+00	 1.6402325e-01	[ 1.1724185e+00]	 2.0538068e-01
      42	 1.2015727e+00	 1.3640374e-01	 1.2476713e+00	 1.6426647e-01	[ 1.1802064e+00]	 1.9459391e-01


.. parsed-literal::

      43	 1.2138922e+00	 1.3584199e-01	 1.2603320e+00	 1.6556482e-01	[ 1.1841992e+00]	 2.1540523e-01


.. parsed-literal::

      44	 1.2245559e+00	 1.3606585e-01	 1.2710338e+00	 1.6505923e-01	[ 1.1913106e+00]	 2.2157693e-01
      45	 1.2303337e+00	 1.3563591e-01	 1.2767862e+00	 1.6477303e-01	[ 1.1978265e+00]	 1.7964172e-01


.. parsed-literal::

      46	 1.2443521e+00	 1.3461401e-01	 1.2912123e+00	 1.6452501e-01	[ 1.2096221e+00]	 2.1506453e-01


.. parsed-literal::

      47	 1.2526667e+00	 1.3499204e-01	 1.2999652e+00	 1.6497357e-01	[ 1.2238333e+00]	 2.1457386e-01
      48	 1.2619715e+00	 1.3473370e-01	 1.3092151e+00	 1.6490824e-01	[ 1.2290789e+00]	 2.0461965e-01


.. parsed-literal::

      49	 1.2700591e+00	 1.3439942e-01	 1.3173751e+00	 1.6437397e-01	[ 1.2317818e+00]	 1.9731665e-01


.. parsed-literal::

      50	 1.2773581e+00	 1.3416475e-01	 1.3247948e+00	 1.6417847e-01	[ 1.2353026e+00]	 2.0808768e-01
      51	 1.2859181e+00	 1.3331384e-01	 1.3335079e+00	 1.6305402e-01	[ 1.2375921e+00]	 1.8512344e-01


.. parsed-literal::

      52	 1.2936627e+00	 1.3356070e-01	 1.3411251e+00	 1.6262054e-01	[ 1.2478574e+00]	 2.0700264e-01


.. parsed-literal::

      53	 1.2984619e+00	 1.3350629e-01	 1.3458960e+00	 1.6238498e-01	[ 1.2548429e+00]	 2.0634365e-01
      54	 1.3072088e+00	 1.3345151e-01	 1.3548691e+00	 1.6162856e-01	[ 1.2650239e+00]	 1.8248320e-01


.. parsed-literal::

      55	 1.3136698e+00	 1.3317066e-01	 1.3616429e+00	 1.6204969e-01	[ 1.2722918e+00]	 1.9862723e-01


.. parsed-literal::

      56	 1.3215607e+00	 1.3303291e-01	 1.3696006e+00	 1.6159632e-01	[ 1.2785044e+00]	 2.1646357e-01


.. parsed-literal::

      57	 1.3274504e+00	 1.3281562e-01	 1.3756575e+00	 1.6169190e-01	[ 1.2816887e+00]	 2.1112680e-01
      58	 1.3326478e+00	 1.3250389e-01	 1.3810385e+00	 1.6183425e-01	[ 1.2849786e+00]	 2.0472717e-01


.. parsed-literal::

      59	 1.3401300e+00	 1.3200047e-01	 1.3888441e+00	 1.6103452e-01	[ 1.2887461e+00]	 2.1069717e-01


.. parsed-literal::

      60	 1.3463607e+00	 1.3185011e-01	 1.3951376e+00	 1.6086295e-01	[ 1.2929806e+00]	 2.1907234e-01
      61	 1.3513614e+00	 1.3128248e-01	 1.4000692e+00	 1.5997381e-01	[ 1.2973297e+00]	 1.8591380e-01


.. parsed-literal::

      62	 1.3583649e+00	 1.3046249e-01	 1.4071364e+00	 1.5903491e-01	[ 1.2989134e+00]	 1.9206715e-01
      63	 1.3630121e+00	 1.2991928e-01	 1.4118860e+00	 1.5848326e-01	  1.2979254e+00 	 2.0514131e-01


.. parsed-literal::

      64	 1.3675597e+00	 1.2949497e-01	 1.4163475e+00	 1.5809042e-01	[ 1.3040691e+00]	 2.0969296e-01


.. parsed-literal::

      65	 1.3746908e+00	 1.2911084e-01	 1.4235399e+00	 1.5818541e-01	[ 1.3123562e+00]	 2.1328115e-01


.. parsed-literal::

      66	 1.3795985e+00	 1.2852456e-01	 1.4287366e+00	 1.5681377e-01	[ 1.3179570e+00]	 2.0741677e-01


.. parsed-literal::

      67	 1.3843051e+00	 1.2834061e-01	 1.4335761e+00	 1.5618612e-01	[ 1.3220260e+00]	 2.0694685e-01


.. parsed-literal::

      68	 1.3891160e+00	 1.2813218e-01	 1.4385902e+00	 1.5557925e-01	[ 1.3232682e+00]	 2.1279407e-01
      69	 1.3936486e+00	 1.2804391e-01	 1.4432285e+00	 1.5480171e-01	[ 1.3303033e+00]	 2.0107651e-01


.. parsed-literal::

      70	 1.3977654e+00	 1.2775188e-01	 1.4472766e+00	 1.5487880e-01	[ 1.3320297e+00]	 2.0804381e-01


.. parsed-literal::

      71	 1.4015703e+00	 1.2743088e-01	 1.4509911e+00	 1.5513601e-01	[ 1.3340779e+00]	 2.1549010e-01


.. parsed-literal::

      72	 1.4042514e+00	 1.2721470e-01	 1.4536217e+00	 1.5548769e-01	[ 1.3357628e+00]	 2.2902226e-01
      73	 1.4078095e+00	 1.2711254e-01	 1.4571456e+00	 1.5587357e-01	[ 1.3405318e+00]	 1.9099236e-01


.. parsed-literal::

      74	 1.4122169e+00	 1.2692755e-01	 1.4616395e+00	 1.5591407e-01	[ 1.3446453e+00]	 2.0975184e-01
      75	 1.4166646e+00	 1.2660604e-01	 1.4663010e+00	 1.5598790e-01	[ 1.3488617e+00]	 1.9883728e-01


.. parsed-literal::

      76	 1.4205611e+00	 1.2640309e-01	 1.4702812e+00	 1.5563848e-01	[ 1.3510898e+00]	 2.1342874e-01


.. parsed-literal::

      77	 1.4248378e+00	 1.2594707e-01	 1.4746342e+00	 1.5520384e-01	  1.3507844e+00 	 2.0266247e-01
      78	 1.4297496e+00	 1.2537219e-01	 1.4796364e+00	 1.5489398e-01	  1.3500535e+00 	 1.7883801e-01


.. parsed-literal::

      79	 1.4344324e+00	 1.2468904e-01	 1.4843647e+00	 1.5482581e-01	  1.3478245e+00 	 1.9425344e-01


.. parsed-literal::

      80	 1.4379621e+00	 1.2431180e-01	 1.4878462e+00	 1.5492350e-01	  1.3504583e+00 	 2.0950246e-01
      81	 1.4414107e+00	 1.2389826e-01	 1.4913118e+00	 1.5522583e-01	[ 1.3520069e+00]	 1.7680311e-01


.. parsed-literal::

      82	 1.4444293e+00	 1.2349229e-01	 1.4945320e+00	 1.5528422e-01	[ 1.3559636e+00]	 2.0890260e-01


.. parsed-literal::

      83	 1.4480407e+00	 1.2287671e-01	 1.4983170e+00	 1.5508415e-01	  1.3526670e+00 	 2.1202993e-01
      84	 1.4508988e+00	 1.2254337e-01	 1.5012560e+00	 1.5477054e-01	  1.3528130e+00 	 1.7664218e-01


.. parsed-literal::

      85	 1.4544223e+00	 1.2191274e-01	 1.5049396e+00	 1.5364469e-01	  1.3514771e+00 	 2.0562458e-01


.. parsed-literal::

      86	 1.4573321e+00	 1.2154435e-01	 1.5078302e+00	 1.5301603e-01	  1.3509951e+00 	 2.1070528e-01


.. parsed-literal::

      87	 1.4601489e+00	 1.2130586e-01	 1.5105722e+00	 1.5253957e-01	  1.3530696e+00 	 2.0265532e-01


.. parsed-literal::

      88	 1.4639538e+00	 1.2097207e-01	 1.5142783e+00	 1.5168858e-01	  1.3515588e+00 	 2.2079897e-01
      89	 1.4653812e+00	 1.2077650e-01	 1.5157931e+00	 1.5098853e-01	  1.3538380e+00 	 1.9006085e-01


.. parsed-literal::

      90	 1.4676464e+00	 1.2071269e-01	 1.5180007e+00	 1.5108842e-01	  1.3544554e+00 	 1.8855739e-01


.. parsed-literal::

      91	 1.4697189e+00	 1.2055817e-01	 1.5201310e+00	 1.5090767e-01	  1.3529645e+00 	 2.0366073e-01


.. parsed-literal::

      92	 1.4716176e+00	 1.2033360e-01	 1.5221105e+00	 1.5061616e-01	  1.3515682e+00 	 2.1240783e-01
      93	 1.4749835e+00	 1.1995206e-01	 1.5255631e+00	 1.4993795e-01	  1.3475709e+00 	 1.9863868e-01


.. parsed-literal::

      94	 1.4767934e+00	 1.1951840e-01	 1.5275005e+00	 1.4923219e-01	  1.3401393e+00 	 2.1560001e-01


.. parsed-literal::

      95	 1.4797605e+00	 1.1948062e-01	 1.5303243e+00	 1.4923222e-01	  1.3458475e+00 	 2.0261240e-01
      96	 1.4813022e+00	 1.1943716e-01	 1.5317855e+00	 1.4912605e-01	  1.3470092e+00 	 1.8409491e-01


.. parsed-literal::

      97	 1.4836827e+00	 1.1928093e-01	 1.5341292e+00	 1.4884880e-01	  1.3468502e+00 	 2.0768499e-01


.. parsed-literal::

      98	 1.4852613e+00	 1.1919864e-01	 1.5357958e+00	 1.4843819e-01	  1.3331559e+00 	 2.1066022e-01
      99	 1.4881527e+00	 1.1906893e-01	 1.5386470e+00	 1.4830708e-01	  1.3389028e+00 	 1.9317436e-01


.. parsed-literal::

     100	 1.4896109e+00	 1.1897124e-01	 1.5401495e+00	 1.4818243e-01	  1.3396243e+00 	 2.1034861e-01


.. parsed-literal::

     101	 1.4915678e+00	 1.1885640e-01	 1.5421748e+00	 1.4797697e-01	  1.3380631e+00 	 2.0986652e-01


.. parsed-literal::

     102	 1.4937889e+00	 1.1858188e-01	 1.5445459e+00	 1.4758575e-01	  1.3348070e+00 	 2.1468425e-01


.. parsed-literal::

     103	 1.4961476e+00	 1.1855990e-01	 1.5468850e+00	 1.4754915e-01	  1.3334952e+00 	 2.0236182e-01


.. parsed-literal::

     104	 1.4973665e+00	 1.1850704e-01	 1.5480659e+00	 1.4764866e-01	  1.3356550e+00 	 2.1975350e-01


.. parsed-literal::

     105	 1.4995546e+00	 1.1831395e-01	 1.5502576e+00	 1.4763493e-01	  1.3359674e+00 	 2.0364785e-01
     106	 1.5016850e+00	 1.1806275e-01	 1.5525140e+00	 1.4790970e-01	  1.3322469e+00 	 1.9167352e-01


.. parsed-literal::

     107	 1.5038743e+00	 1.1773930e-01	 1.5547675e+00	 1.4758297e-01	  1.3328697e+00 	 2.0077777e-01


.. parsed-literal::

     108	 1.5054714e+00	 1.1756211e-01	 1.5564117e+00	 1.4730774e-01	  1.3313275e+00 	 2.1490073e-01
     109	 1.5072296e+00	 1.1729712e-01	 1.5582837e+00	 1.4697967e-01	  1.3276894e+00 	 1.9790673e-01


.. parsed-literal::

     110	 1.5090137e+00	 1.1700467e-01	 1.5601527e+00	 1.4668041e-01	  1.3246000e+00 	 2.0313120e-01


.. parsed-literal::

     111	 1.5107921e+00	 1.1680459e-01	 1.5619268e+00	 1.4662787e-01	  1.3281197e+00 	 2.1826005e-01


.. parsed-literal::

     112	 1.5124207e+00	 1.1655433e-01	 1.5635636e+00	 1.4658411e-01	  1.3274232e+00 	 2.1245098e-01


.. parsed-literal::

     113	 1.5134565e+00	 1.1643945e-01	 1.5645570e+00	 1.4661826e-01	  1.3288566e+00 	 2.0916343e-01


.. parsed-literal::

     114	 1.5158097e+00	 1.1610455e-01	 1.5668096e+00	 1.4651711e-01	  1.3288021e+00 	 2.0920467e-01


.. parsed-literal::

     115	 1.5177616e+00	 1.1585361e-01	 1.5687629e+00	 1.4632035e-01	  1.3295579e+00 	 2.1181583e-01


.. parsed-literal::

     116	 1.5191124e+00	 1.1561406e-01	 1.5701876e+00	 1.4588717e-01	  1.3296449e+00 	 2.0674539e-01


.. parsed-literal::

     117	 1.5205838e+00	 1.1560692e-01	 1.5716634e+00	 1.4583369e-01	  1.3309769e+00 	 2.1773338e-01
     118	 1.5216536e+00	 1.1560328e-01	 1.5727998e+00	 1.4579564e-01	  1.3318655e+00 	 1.8693471e-01


.. parsed-literal::

     119	 1.5231385e+00	 1.1553875e-01	 1.5743931e+00	 1.4579317e-01	  1.3306328e+00 	 1.8542719e-01
     120	 1.5248952e+00	 1.1542037e-01	 1.5763118e+00	 1.4575351e-01	  1.3269136e+00 	 1.9953108e-01


.. parsed-literal::

     121	 1.5266257e+00	 1.1530799e-01	 1.5780754e+00	 1.4576881e-01	  1.3235691e+00 	 2.0955896e-01


.. parsed-literal::

     122	 1.5277747e+00	 1.1521797e-01	 1.5791534e+00	 1.4574156e-01	  1.3221774e+00 	 2.1195555e-01


.. parsed-literal::

     123	 1.5289825e+00	 1.1504999e-01	 1.5802962e+00	 1.4570699e-01	  1.3196747e+00 	 2.1985149e-01


.. parsed-literal::

     124	 1.5298116e+00	 1.1489701e-01	 1.5811444e+00	 1.4553717e-01	  1.3103100e+00 	 2.1043658e-01


.. parsed-literal::

     125	 1.5307062e+00	 1.1486297e-01	 1.5820315e+00	 1.4550238e-01	  1.3119995e+00 	 2.1594596e-01


.. parsed-literal::

     126	 1.5320809e+00	 1.1472558e-01	 1.5834646e+00	 1.4534375e-01	  1.3111705e+00 	 2.1255279e-01
     127	 1.5330774e+00	 1.1460038e-01	 1.5844793e+00	 1.4522144e-01	  1.3117496e+00 	 1.9462657e-01


.. parsed-literal::

     128	 1.5352217e+00	 1.1427325e-01	 1.5866927e+00	 1.4515364e-01	  1.3118114e+00 	 2.1620512e-01


.. parsed-literal::

     129	 1.5357909e+00	 1.1409673e-01	 1.5873426e+00	 1.4523017e-01	  1.3091066e+00 	 2.0909452e-01


.. parsed-literal::

     130	 1.5373084e+00	 1.1409793e-01	 1.5887841e+00	 1.4521753e-01	  1.3123791e+00 	 2.0615411e-01
     131	 1.5380405e+00	 1.1406724e-01	 1.5895093e+00	 1.4527436e-01	  1.3119251e+00 	 2.0035601e-01


.. parsed-literal::

     132	 1.5392042e+00	 1.1397057e-01	 1.5907178e+00	 1.4536954e-01	  1.3099785e+00 	 2.1048260e-01


.. parsed-literal::

     133	 1.5404905e+00	 1.1383109e-01	 1.5920554e+00	 1.4553585e-01	  1.3067263e+00 	 2.1288013e-01
     134	 1.5420507e+00	 1.1369270e-01	 1.5936630e+00	 1.4563841e-01	  1.3060596e+00 	 1.8766546e-01


.. parsed-literal::

     135	 1.5435170e+00	 1.1354064e-01	 1.5951754e+00	 1.4571058e-01	  1.3071035e+00 	 1.8941522e-01


.. parsed-literal::

     136	 1.5445002e+00	 1.1342079e-01	 1.5961747e+00	 1.4580346e-01	  1.3115838e+00 	 2.1817446e-01


.. parsed-literal::

     137	 1.5454021e+00	 1.1337680e-01	 1.5970544e+00	 1.4575868e-01	  1.3122466e+00 	 2.1148300e-01
     138	 1.5461774e+00	 1.1335207e-01	 1.5978015e+00	 1.4573049e-01	  1.3131726e+00 	 1.9761825e-01


.. parsed-literal::

     139	 1.5470914e+00	 1.1328701e-01	 1.5987126e+00	 1.4570961e-01	  1.3135648e+00 	 2.0747685e-01


.. parsed-literal::

     140	 1.5486393e+00	 1.1313363e-01	 1.6002967e+00	 1.4553297e-01	  1.3124051e+00 	 2.1280980e-01


.. parsed-literal::

     141	 1.5496520e+00	 1.1301779e-01	 1.6013847e+00	 1.4549668e-01	  1.3100173e+00 	 3.1676388e-01
     142	 1.5509165e+00	 1.1291151e-01	 1.6026790e+00	 1.4525299e-01	  1.3080314e+00 	 1.9585037e-01


.. parsed-literal::

     143	 1.5520041e+00	 1.1281332e-01	 1.6038240e+00	 1.4506232e-01	  1.3049346e+00 	 2.0891142e-01
     144	 1.5534144e+00	 1.1269777e-01	 1.6053005e+00	 1.4487924e-01	  1.3013003e+00 	 1.8261766e-01


.. parsed-literal::

     145	 1.5545870e+00	 1.1250181e-01	 1.6066610e+00	 1.4460834e-01	  1.2930410e+00 	 2.1009755e-01


.. parsed-literal::

     146	 1.5560795e+00	 1.1245087e-01	 1.6081212e+00	 1.4463555e-01	  1.2954059e+00 	 2.0528746e-01


.. parsed-literal::

     147	 1.5567846e+00	 1.1244707e-01	 1.6087893e+00	 1.4467573e-01	  1.2978425e+00 	 2.1108866e-01


.. parsed-literal::

     148	 1.5578150e+00	 1.1238694e-01	 1.6098143e+00	 1.4470223e-01	  1.2998308e+00 	 2.0760751e-01


.. parsed-literal::

     149	 1.5583763e+00	 1.1230157e-01	 1.6104407e+00	 1.4455125e-01	  1.2980503e+00 	 2.2007823e-01


.. parsed-literal::

     150	 1.5595858e+00	 1.1225754e-01	 1.6116105e+00	 1.4448404e-01	  1.3006626e+00 	 2.1049452e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.1 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f600c0d8ee0>



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
    CPU times: user 2.11 s, sys: 34 ms, total: 2.14 s
    Wall time: 645 ms


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

