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
       1	-3.4538497e-01	 3.2137678e-01	-3.3560425e-01	 3.1756984e-01	[-3.2778450e-01]	 4.5787668e-01


.. parsed-literal::

       2	-2.7484458e-01	 3.1050894e-01	-2.5057277e-01	 3.0714108e-01	[-2.4059434e-01]	 2.2910953e-01


.. parsed-literal::

       3	-2.3173165e-01	 2.9065633e-01	-1.9025659e-01	 2.8783157e-01	[-1.7997245e-01]	 2.7718043e-01


.. parsed-literal::

       4	-1.9707222e-01	 2.6614466e-01	-1.5424472e-01	 2.6778448e-01	[-1.6071566e-01]	 2.1343184e-01


.. parsed-literal::

       5	-1.0815103e-01	 2.5722710e-01	-7.2304119e-02	 2.5901564e-01	[-8.2500724e-02]	 2.0749998e-01


.. parsed-literal::

       6	-6.9910310e-02	 2.5116129e-01	-3.8109972e-02	 2.5367150e-01	[-4.7741442e-02]	 2.0885730e-01
       7	-5.1820723e-02	 2.4844994e-01	-2.6648158e-02	 2.5124199e-01	[-3.8049188e-02]	 1.8878174e-01


.. parsed-literal::

       8	-3.7355689e-02	 2.4600433e-01	-1.6473262e-02	 2.4900782e-01	[-2.9178273e-02]	 2.0277905e-01
       9	-2.2893719e-02	 2.4335875e-01	-5.1695316e-03	 2.4754554e-01	[-2.3391034e-02]	 1.7392468e-01


.. parsed-literal::

      10	-1.1815841e-02	 2.4131678e-01	 3.7113186e-03	 2.4643831e-01	[-1.7853866e-02]	 2.1547270e-01
      11	-8.6111866e-03	 2.4072857e-01	 5.5070534e-03	 2.4542299e-01	[-1.7057768e-02]	 1.9978452e-01


.. parsed-literal::

      12	-4.3175696e-03	 2.4012301e-01	 9.6748591e-03	 2.4488860e-01	[-1.1227703e-02]	 1.8661690e-01


.. parsed-literal::

      13	-2.0743209e-03	 2.3964548e-01	 1.1896781e-02	 2.4446533e-01	[-9.1148380e-03]	 2.1757078e-01
      14	 2.5583509e-03	 2.3870226e-01	 1.6862246e-02	 2.4353614e-01	[-4.1489887e-03]	 2.0062065e-01


.. parsed-literal::

      15	 1.0961706e-01	 2.2395115e-01	 1.2948035e-01	 2.3017715e-01	[ 1.1277304e-01]	 3.0236077e-01


.. parsed-literal::

      16	 1.8722663e-01	 2.1820591e-01	 2.1042637e-01	 2.2589121e-01	[ 1.9370561e-01]	 2.0651889e-01
      17	 2.7829246e-01	 2.1708396e-01	 3.0982865e-01	 2.3048424e-01	[ 2.7357053e-01]	 1.7703462e-01


.. parsed-literal::

      18	 3.3820746e-01	 2.1268954e-01	 3.7046024e-01	 2.2059297e-01	[ 3.4142888e-01]	 2.0214343e-01


.. parsed-literal::

      19	 3.7990779e-01	 2.0721995e-01	 4.1262057e-01	 2.1505675e-01	[ 3.8972269e-01]	 2.0683336e-01


.. parsed-literal::

      20	 4.3240608e-01	 2.0041164e-01	 4.6590374e-01	 2.1033112e-01	[ 4.3587114e-01]	 2.0220923e-01


.. parsed-literal::

      21	 4.9133202e-01	 1.9519027e-01	 5.2592544e-01	 2.0582752e-01	[ 4.9544181e-01]	 2.1417260e-01
      22	 6.0235813e-01	 1.9182118e-01	 6.3971916e-01	 2.0271233e-01	[ 6.2718289e-01]	 2.0335770e-01


.. parsed-literal::

      23	 6.5123327e-01	 1.8505867e-01	 6.8970223e-01	 1.9797524e-01	[ 6.8293957e-01]	 2.0889688e-01
      24	 6.9009850e-01	 1.8270704e-01	 7.2928922e-01	 1.9660475e-01	[ 7.2412233e-01]	 1.9032598e-01


.. parsed-literal::

      25	 7.3620581e-01	 1.7795355e-01	 7.7316006e-01	 1.9245603e-01	[ 7.7236306e-01]	 2.1053505e-01


.. parsed-literal::

      26	 7.6429693e-01	 1.8419438e-01	 8.0274967e-01	 1.9807418e-01	[ 7.9138866e-01]	 2.0718503e-01
      27	 8.0761110e-01	 1.7628352e-01	 8.4687831e-01	 1.9448403e-01	[ 8.3870239e-01]	 1.7944670e-01


.. parsed-literal::

      28	 8.2656969e-01	 1.7118503e-01	 8.6559654e-01	 1.8894528e-01	[ 8.5853530e-01]	 2.0551610e-01
      29	 8.5060794e-01	 1.6949952e-01	 8.9025425e-01	 1.8622814e-01	[ 8.8584011e-01]	 1.7510915e-01


.. parsed-literal::

      30	 8.7547921e-01	 1.6698282e-01	 9.1541002e-01	 1.8369994e-01	[ 9.0909482e-01]	 2.1705317e-01


.. parsed-literal::

      31	 9.0051513e-01	 1.6593662e-01	 9.4138899e-01	 1.8430598e-01	[ 9.2931620e-01]	 2.1297288e-01


.. parsed-literal::

      32	 9.1787926e-01	 1.6547218e-01	 9.5935957e-01	 1.8397348e-01	[ 9.4712776e-01]	 2.1510935e-01


.. parsed-literal::

      33	 9.3696901e-01	 1.6386856e-01	 9.7902483e-01	 1.8290768e-01	[ 9.6553500e-01]	 2.0899391e-01


.. parsed-literal::

      34	 9.5493451e-01	 1.6133519e-01	 9.9752148e-01	 1.8047806e-01	[ 9.8330825e-01]	 2.1205950e-01
      35	 9.7141431e-01	 1.5952493e-01	 1.0146784e+00	 1.7834606e-01	[ 9.9816137e-01]	 1.9946361e-01


.. parsed-literal::

      36	 9.8767411e-01	 1.5796629e-01	 1.0316319e+00	 1.7660361e-01	[ 1.0076352e+00]	 2.1082973e-01


.. parsed-literal::

      37	 1.0036186e+00	 1.5698234e-01	 1.0480946e+00	 1.7570201e-01	[ 1.0189145e+00]	 2.0644450e-01


.. parsed-literal::

      38	 1.0254733e+00	 1.5515881e-01	 1.0709845e+00	 1.7345149e-01	[ 1.0336364e+00]	 2.0939898e-01
      39	 1.0402404e+00	 1.5424159e-01	 1.0869551e+00	 1.7332736e-01	[ 1.0355038e+00]	 1.7348194e-01


.. parsed-literal::

      40	 1.0560635e+00	 1.5193338e-01	 1.1025103e+00	 1.7092747e-01	[ 1.0537779e+00]	 2.1700835e-01
      41	 1.0665897e+00	 1.5106058e-01	 1.1130721e+00	 1.6964295e-01	[ 1.0671181e+00]	 1.9376779e-01


.. parsed-literal::

      42	 1.0768423e+00	 1.5012215e-01	 1.1238363e+00	 1.6818631e-01	[ 1.0753966e+00]	 1.9864202e-01


.. parsed-literal::

      43	 1.0911331e+00	 1.4915800e-01	 1.1380688e+00	 1.6752551e-01	[ 1.0949363e+00]	 2.0485139e-01


.. parsed-literal::

      44	 1.1027266e+00	 1.4776218e-01	 1.1500072e+00	 1.6622713e-01	[ 1.1035152e+00]	 2.0605588e-01
      45	 1.1124889e+00	 1.4671827e-01	 1.1599032e+00	 1.6643852e-01	[ 1.1136564e+00]	 1.9743156e-01


.. parsed-literal::

      46	 1.1230858e+00	 1.4585869e-01	 1.1705702e+00	 1.6680807e-01	[ 1.1247709e+00]	 2.0157766e-01
      47	 1.1338174e+00	 1.4537025e-01	 1.1815010e+00	 1.6872643e-01	  1.1240432e+00 	 1.8211913e-01


.. parsed-literal::

      48	 1.1438127e+00	 1.4483433e-01	 1.1915389e+00	 1.6843337e-01	[ 1.1305491e+00]	 2.1163678e-01


.. parsed-literal::

      49	 1.1540874e+00	 1.4416191e-01	 1.2021851e+00	 1.6750621e-01	[ 1.1375463e+00]	 2.0261884e-01
      50	 1.1642532e+00	 1.4278691e-01	 1.2125138e+00	 1.6606881e-01	[ 1.1418110e+00]	 1.9709349e-01


.. parsed-literal::

      51	 1.1754356e+00	 1.4156152e-01	 1.2240317e+00	 1.6469844e-01	[ 1.1478321e+00]	 2.0114374e-01


.. parsed-literal::

      52	 1.1855576e+00	 1.4022413e-01	 1.2343834e+00	 1.6382559e-01	[ 1.1516012e+00]	 2.0754886e-01


.. parsed-literal::

      53	 1.1984494e+00	 1.3856750e-01	 1.2475687e+00	 1.6356260e-01	[ 1.1563283e+00]	 2.0381498e-01


.. parsed-literal::

      54	 1.2087246e+00	 1.3770946e-01	 1.2580161e+00	 1.6323827e-01	[ 1.1687664e+00]	 2.0965433e-01


.. parsed-literal::

      55	 1.2198293e+00	 1.3715138e-01	 1.2690512e+00	 1.6358813e-01	  1.1686994e+00 	 2.1026897e-01


.. parsed-literal::

      56	 1.2277835e+00	 1.3685075e-01	 1.2770967e+00	 1.6356343e-01	  1.1687049e+00 	 2.1334791e-01


.. parsed-literal::

      57	 1.2354001e+00	 1.3623373e-01	 1.2851048e+00	 1.6372861e-01	  1.1588067e+00 	 2.1390295e-01


.. parsed-literal::

      58	 1.2443723e+00	 1.3593420e-01	 1.2942532e+00	 1.6297710e-01	  1.1558182e+00 	 2.1039438e-01


.. parsed-literal::

      59	 1.2512357e+00	 1.3537874e-01	 1.3010499e+00	 1.6255043e-01	  1.1586386e+00 	 2.0889544e-01


.. parsed-literal::

      60	 1.2613942e+00	 1.3436335e-01	 1.3115900e+00	 1.6119587e-01	  1.1585843e+00 	 2.0418978e-01
      61	 1.2672701e+00	 1.3366081e-01	 1.3175997e+00	 1.6148809e-01	  1.1638853e+00 	 1.9844079e-01


.. parsed-literal::

      62	 1.2755288e+00	 1.3330022e-01	 1.3259049e+00	 1.6132068e-01	[ 1.1725879e+00]	 1.7845678e-01


.. parsed-literal::

      63	 1.2859520e+00	 1.3289868e-01	 1.3367684e+00	 1.6172122e-01	[ 1.1799685e+00]	 2.0871902e-01


.. parsed-literal::

      64	 1.2923122e+00	 1.3252245e-01	 1.3432501e+00	 1.6205081e-01	[ 1.1897567e+00]	 2.1327591e-01
      65	 1.3031980e+00	 1.3195701e-01	 1.3544362e+00	 1.6234981e-01	[ 1.2131434e+00]	 1.9887638e-01


.. parsed-literal::

      66	 1.3098827e+00	 1.3204265e-01	 1.3608563e+00	 1.6319172e-01	[ 1.2177018e+00]	 1.8451548e-01
      67	 1.3142161e+00	 1.3178636e-01	 1.3649435e+00	 1.6251471e-01	[ 1.2227234e+00]	 1.8866587e-01


.. parsed-literal::

      68	 1.3197639e+00	 1.3167790e-01	 1.3703909e+00	 1.6226364e-01	[ 1.2284043e+00]	 2.1742344e-01


.. parsed-literal::

      69	 1.3251305e+00	 1.3166167e-01	 1.3757431e+00	 1.6223733e-01	[ 1.2309264e+00]	 2.0974016e-01


.. parsed-literal::

      70	 1.3317021e+00	 1.3138886e-01	 1.3825122e+00	 1.6188840e-01	  1.2261165e+00 	 2.1321702e-01
      71	 1.3381601e+00	 1.3140577e-01	 1.3890764e+00	 1.6213930e-01	  1.2257440e+00 	 1.9338155e-01


.. parsed-literal::

      72	 1.3432916e+00	 1.3081620e-01	 1.3943785e+00	 1.6204150e-01	  1.2209476e+00 	 2.0096827e-01


.. parsed-literal::

      73	 1.3497986e+00	 1.3052724e-01	 1.4012804e+00	 1.6197102e-01	  1.2127443e+00 	 2.0712352e-01


.. parsed-literal::

      74	 1.3559298e+00	 1.2984929e-01	 1.4080294e+00	 1.6169311e-01	  1.2046289e+00 	 2.0527959e-01


.. parsed-literal::

      75	 1.3621770e+00	 1.2979328e-01	 1.4142957e+00	 1.6087896e-01	  1.2060031e+00 	 2.0385551e-01


.. parsed-literal::

      76	 1.3667616e+00	 1.2992838e-01	 1.4189347e+00	 1.6009803e-01	  1.2100052e+00 	 2.0775175e-01


.. parsed-literal::

      77	 1.3706464e+00	 1.2971098e-01	 1.4229551e+00	 1.5890768e-01	  1.2131677e+00 	 2.1525526e-01
      78	 1.3757599e+00	 1.2904695e-01	 1.4282221e+00	 1.5742204e-01	  1.2166363e+00 	 1.9903278e-01


.. parsed-literal::

      79	 1.3805118e+00	 1.2808431e-01	 1.4330347e+00	 1.5659416e-01	  1.2214319e+00 	 2.0758986e-01


.. parsed-literal::

      80	 1.3835298e+00	 1.2713272e-01	 1.4361114e+00	 1.5557427e-01	  1.2201267e+00 	 2.0680547e-01


.. parsed-literal::

      81	 1.3867812e+00	 1.2690417e-01	 1.4391733e+00	 1.5611803e-01	  1.2242255e+00 	 2.0737028e-01


.. parsed-literal::

      82	 1.3892830e+00	 1.2683724e-01	 1.4416336e+00	 1.5648233e-01	  1.2276714e+00 	 2.1721601e-01


.. parsed-literal::

      83	 1.3926923e+00	 1.2664659e-01	 1.4449292e+00	 1.5695467e-01	[ 1.2334486e+00]	 2.1633363e-01
      84	 1.3984068e+00	 1.2637325e-01	 1.4506521e+00	 1.5747282e-01	[ 1.2413961e+00]	 1.9083381e-01


.. parsed-literal::

      85	 1.4033764e+00	 1.2593819e-01	 1.4557995e+00	 1.5781772e-01	[ 1.2540059e+00]	 1.8622351e-01


.. parsed-literal::

      86	 1.4082362e+00	 1.2583882e-01	 1.4605762e+00	 1.5744503e-01	[ 1.2594447e+00]	 2.0975876e-01


.. parsed-literal::

      87	 1.4115094e+00	 1.2560577e-01	 1.4639073e+00	 1.5691538e-01	[ 1.2628817e+00]	 2.0254016e-01


.. parsed-literal::

      88	 1.4153311e+00	 1.2532339e-01	 1.4679386e+00	 1.5614800e-01	[ 1.2673209e+00]	 2.0537281e-01
      89	 1.4169485e+00	 1.2491587e-01	 1.4697884e+00	 1.5586684e-01	  1.2665584e+00 	 1.9721007e-01


.. parsed-literal::

      90	 1.4200233e+00	 1.2484939e-01	 1.4727768e+00	 1.5588763e-01	[ 1.2710436e+00]	 1.9809008e-01


.. parsed-literal::

      91	 1.4222142e+00	 1.2473777e-01	 1.4749694e+00	 1.5593979e-01	[ 1.2728667e+00]	 2.0592856e-01


.. parsed-literal::

      92	 1.4246724e+00	 1.2458378e-01	 1.4775251e+00	 1.5593773e-01	[ 1.2738886e+00]	 2.0159101e-01


.. parsed-literal::

      93	 1.4293633e+00	 1.2435981e-01	 1.4824449e+00	 1.5598814e-01	  1.2734111e+00 	 2.0736885e-01


.. parsed-literal::

      94	 1.4321307e+00	 1.2440066e-01	 1.4853946e+00	 1.5625491e-01	  1.2721837e+00 	 3.1285787e-01


.. parsed-literal::

      95	 1.4360030e+00	 1.2435954e-01	 1.4894332e+00	 1.5629672e-01	  1.2699437e+00 	 2.0236588e-01


.. parsed-literal::

      96	 1.4382775e+00	 1.2427553e-01	 1.4917229e+00	 1.5645559e-01	  1.2714584e+00 	 2.0897841e-01
      97	 1.4408212e+00	 1.2441302e-01	 1.4942874e+00	 1.5680891e-01	  1.2687114e+00 	 2.0000529e-01


.. parsed-literal::

      98	 1.4436257e+00	 1.2427590e-01	 1.4970770e+00	 1.5758778e-01	  1.2685512e+00 	 2.1565151e-01


.. parsed-literal::

      99	 1.4463268e+00	 1.2427645e-01	 1.4997307e+00	 1.5813256e-01	  1.2661157e+00 	 2.1060848e-01
     100	 1.4488683e+00	 1.2398336e-01	 1.5023196e+00	 1.5909841e-01	  1.2554377e+00 	 1.7412114e-01


.. parsed-literal::

     101	 1.4504877e+00	 1.2382714e-01	 1.5039022e+00	 1.5916470e-01	  1.2548185e+00 	 1.7567182e-01
     102	 1.4515989e+00	 1.2359904e-01	 1.5049835e+00	 1.5904633e-01	  1.2558232e+00 	 1.9684958e-01


.. parsed-literal::

     103	 1.4541994e+00	 1.2294298e-01	 1.5076719e+00	 1.5902213e-01	  1.2521218e+00 	 2.1441841e-01


.. parsed-literal::

     104	 1.4556693e+00	 1.2254396e-01	 1.5091876e+00	 1.5922772e-01	  1.2467890e+00 	 2.1268487e-01
     105	 1.4574420e+00	 1.2235907e-01	 1.5109629e+00	 1.5900986e-01	  1.2485221e+00 	 1.9883752e-01


.. parsed-literal::

     106	 1.4601780e+00	 1.2214874e-01	 1.5137439e+00	 1.5871865e-01	  1.2500432e+00 	 1.8763137e-01


.. parsed-literal::

     107	 1.4618233e+00	 1.2204783e-01	 1.5154214e+00	 1.5858340e-01	  1.2499658e+00 	 2.0647240e-01


.. parsed-literal::

     108	 1.4659005e+00	 1.2155860e-01	 1.5196246e+00	 1.5854767e-01	  1.2436498e+00 	 2.0432544e-01


.. parsed-literal::

     109	 1.4676081e+00	 1.2150844e-01	 1.5213748e+00	 1.5801457e-01	  1.2405712e+00 	 3.2907104e-01


.. parsed-literal::

     110	 1.4696946e+00	 1.2124905e-01	 1.5234696e+00	 1.5791583e-01	  1.2394541e+00 	 2.0649862e-01


.. parsed-literal::

     111	 1.4720029e+00	 1.2095427e-01	 1.5257898e+00	 1.5777645e-01	  1.2351944e+00 	 2.2092676e-01


.. parsed-literal::

     112	 1.4740951e+00	 1.2070378e-01	 1.5279351e+00	 1.5758461e-01	  1.2372532e+00 	 2.1237230e-01
     113	 1.4765833e+00	 1.2060750e-01	 1.5304334e+00	 1.5731184e-01	  1.2372480e+00 	 1.7126203e-01


.. parsed-literal::

     114	 1.4786084e+00	 1.2057424e-01	 1.5324734e+00	 1.5718516e-01	  1.2413635e+00 	 2.0782399e-01


.. parsed-literal::

     115	 1.4800732e+00	 1.2048197e-01	 1.5339150e+00	 1.5713919e-01	  1.2430791e+00 	 2.0373321e-01


.. parsed-literal::

     116	 1.4812141e+00	 1.2037555e-01	 1.5350424e+00	 1.5724991e-01	  1.2432851e+00 	 2.1769524e-01


.. parsed-literal::

     117	 1.4835818e+00	 1.2013709e-01	 1.5374125e+00	 1.5760422e-01	  1.2402846e+00 	 2.0842957e-01


.. parsed-literal::

     118	 1.4847577e+00	 1.1998570e-01	 1.5386115e+00	 1.5799418e-01	  1.2363602e+00 	 2.8965807e-01
     119	 1.4864311e+00	 1.1982223e-01	 1.5402948e+00	 1.5835015e-01	  1.2326240e+00 	 1.7345071e-01


.. parsed-literal::

     120	 1.4880822e+00	 1.1967297e-01	 1.5419791e+00	 1.5867555e-01	  1.2282960e+00 	 2.0149946e-01


.. parsed-literal::

     121	 1.4899168e+00	 1.1950360e-01	 1.5438775e+00	 1.5907026e-01	  1.2240574e+00 	 2.1527696e-01


.. parsed-literal::

     122	 1.4910663e+00	 1.1941014e-01	 1.5451487e+00	 1.5933493e-01	  1.2169883e+00 	 2.1253967e-01
     123	 1.4927498e+00	 1.1927057e-01	 1.5468038e+00	 1.5932963e-01	  1.2196738e+00 	 1.8567991e-01


.. parsed-literal::

     124	 1.4939368e+00	 1.1915848e-01	 1.5479842e+00	 1.5919075e-01	  1.2204815e+00 	 2.0772624e-01
     125	 1.4954029e+00	 1.1897736e-01	 1.5494947e+00	 1.5904290e-01	  1.2176606e+00 	 2.0380235e-01


.. parsed-literal::

     126	 1.4974751e+00	 1.1873902e-01	 1.5517094e+00	 1.5868543e-01	  1.2019741e+00 	 2.1542048e-01
     127	 1.4985966e+00	 1.1841502e-01	 1.5530444e+00	 1.5866339e-01	  1.1902625e+00 	 1.9617176e-01


.. parsed-literal::

     128	 1.5000467e+00	 1.1842570e-01	 1.5544088e+00	 1.5871426e-01	  1.1887931e+00 	 1.8690896e-01
     129	 1.5009486e+00	 1.1840320e-01	 1.5552936e+00	 1.5872893e-01	  1.1862580e+00 	 1.9164801e-01


.. parsed-literal::

     130	 1.5028500e+00	 1.1823288e-01	 1.5572591e+00	 1.5859451e-01	  1.1790648e+00 	 2.1666503e-01


.. parsed-literal::

     131	 1.5040559e+00	 1.1804770e-01	 1.5585583e+00	 1.5836991e-01	  1.1670593e+00 	 2.1815062e-01
     132	 1.5061505e+00	 1.1787346e-01	 1.5606960e+00	 1.5815878e-01	  1.1653378e+00 	 1.9596529e-01


.. parsed-literal::

     133	 1.5068802e+00	 1.1783200e-01	 1.5614119e+00	 1.5804177e-01	  1.1666305e+00 	 1.8075228e-01


.. parsed-literal::

     134	 1.5081218e+00	 1.1776177e-01	 1.5626472e+00	 1.5786136e-01	  1.1645372e+00 	 2.1357012e-01
     135	 1.5098841e+00	 1.1767501e-01	 1.5644303e+00	 1.5764168e-01	  1.1546330e+00 	 1.7847300e-01


.. parsed-literal::

     136	 1.5113977e+00	 1.1764447e-01	 1.5659197e+00	 1.5774699e-01	  1.1510611e+00 	 2.0190835e-01


.. parsed-literal::

     137	 1.5130601e+00	 1.1761143e-01	 1.5676149e+00	 1.5791762e-01	  1.1412516e+00 	 2.0463586e-01


.. parsed-literal::

     138	 1.5144896e+00	 1.1747844e-01	 1.5691082e+00	 1.5815738e-01	  1.1311412e+00 	 2.0890236e-01
     139	 1.5157815e+00	 1.1739530e-01	 1.5704656e+00	 1.5836737e-01	  1.1249037e+00 	 1.7945957e-01


.. parsed-literal::

     140	 1.5172487e+00	 1.1726870e-01	 1.5720015e+00	 1.5820413e-01	  1.1140924e+00 	 2.0727921e-01


.. parsed-literal::

     141	 1.5186362e+00	 1.1708564e-01	 1.5734455e+00	 1.5828268e-01	  1.1009262e+00 	 2.1064401e-01


.. parsed-literal::

     142	 1.5198720e+00	 1.1705472e-01	 1.5746570e+00	 1.5809679e-01	  1.0973857e+00 	 2.1036386e-01


.. parsed-literal::

     143	 1.5214415e+00	 1.1702903e-01	 1.5762010e+00	 1.5800801e-01	  1.0874296e+00 	 2.0599508e-01
     144	 1.5227914e+00	 1.1697354e-01	 1.5775754e+00	 1.5808525e-01	  1.0802991e+00 	 1.9824982e-01


.. parsed-literal::

     145	 1.5241626e+00	 1.1690471e-01	 1.5789839e+00	 1.5810848e-01	  1.0687121e+00 	 2.1005464e-01
     146	 1.5251791e+00	 1.1682652e-01	 1.5800354e+00	 1.5819422e-01	  1.0608166e+00 	 1.8558049e-01


.. parsed-literal::

     147	 1.5261367e+00	 1.1674094e-01	 1.5809500e+00	 1.5810719e-01	  1.0647800e+00 	 2.4808741e-01


.. parsed-literal::

     148	 1.5272450e+00	 1.1660923e-01	 1.5820546e+00	 1.5789451e-01	  1.0660694e+00 	 2.0647645e-01


.. parsed-literal::

     149	 1.5285154e+00	 1.1645785e-01	 1.5833305e+00	 1.5754570e-01	  1.0655455e+00 	 2.0901179e-01
     150	 1.5296319e+00	 1.1625607e-01	 1.5845301e+00	 1.5703618e-01	  1.0633873e+00 	 1.9805193e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.08 s, total: 2min 5s
    Wall time: 31.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fae95e77bb0>



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
    CPU times: user 1.79 s, sys: 43.9 ms, total: 1.84 s
    Wall time: 602 ms


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

