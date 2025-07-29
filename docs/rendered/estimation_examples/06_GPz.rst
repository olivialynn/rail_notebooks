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
       1	-3.4774975e-01	 3.2206286e-01	-3.3807910e-01	 3.1453939e-01	[-3.2431448e-01]	 4.7091961e-01


.. parsed-literal::

       2	-2.7688559e-01	 3.1116803e-01	-2.5282542e-01	 3.0509295e-01	[-2.3476205e-01]	 2.3463988e-01


.. parsed-literal::

       3	-2.3464090e-01	 2.9155135e-01	-1.9352493e-01	 2.8595835e-01	[-1.7228310e-01]	 2.8071475e-01
       4	-1.9495109e-01	 2.6765026e-01	-1.5381940e-01	 2.5997891e-01	[-1.1259305e-01]	 2.0175266e-01


.. parsed-literal::

       5	-1.1240321e-01	 2.5876723e-01	-7.9613284e-02	 2.5134591e-01	[-4.6590816e-02]	 2.0316410e-01


.. parsed-literal::

       6	-7.5055349e-02	 2.5289309e-01	-4.5373204e-02	 2.4569286e-01	[-1.9378391e-02]	 2.1657157e-01


.. parsed-literal::

       7	-5.9695841e-02	 2.5061871e-01	-3.5362722e-02	 2.4306107e-01	[-6.3680459e-03]	 2.0326376e-01
       8	-4.4172867e-02	 2.4792604e-01	-2.4157451e-02	 2.4004814e-01	[ 6.6396168e-03]	 1.7672968e-01


.. parsed-literal::

       9	-3.1164897e-02	 2.4547569e-01	-1.3926638e-02	 2.3768336e-01	[ 1.7602664e-02]	 2.1183181e-01
      10	-2.3938987e-02	 2.4431014e-01	-8.8834539e-03	 2.3679178e-01	[ 1.8909865e-02]	 1.9838119e-01


.. parsed-literal::

      11	-1.6723712e-02	 2.4293578e-01	-2.4264413e-03	 2.3568627e-01	[ 2.7144258e-02]	 2.1095705e-01


.. parsed-literal::

      12	-1.4495191e-02	 2.4254154e-01	-3.9614317e-04	 2.3521853e-01	[ 2.8809769e-02]	 2.1505547e-01


.. parsed-literal::

      13	-1.0844548e-02	 2.4188439e-01	 3.0457974e-03	 2.3408615e-01	[ 3.4196329e-02]	 2.0880723e-01


.. parsed-literal::

      14	 2.3121753e-02	 2.3497793e-01	 3.9500779e-02	 2.2665835e-01	[ 6.6580690e-02]	 3.2403255e-01


.. parsed-literal::

      15	 5.0140056e-02	 2.2778217e-01	 6.8403598e-02	 2.2019978e-01	[ 8.6632066e-02]	 3.0769563e-01


.. parsed-literal::

      16	 8.6730162e-02	 2.2185143e-01	 1.0564084e-01	 2.1539732e-01	[ 1.1627889e-01]	 2.1832204e-01
      17	 1.9810624e-01	 2.1668188e-01	 2.2057857e-01	 2.1148033e-01	[ 2.2977867e-01]	 1.8710089e-01


.. parsed-literal::

      18	 2.8653748e-01	 2.1324817e-01	 3.1922038e-01	 2.0972261e-01	[ 3.2342774e-01]	 2.0930004e-01


.. parsed-literal::

      19	 3.3603127e-01	 2.0802989e-01	 3.6734605e-01	 2.0655396e-01	[ 3.6973238e-01]	 2.1944976e-01


.. parsed-literal::

      20	 3.8449997e-01	 2.0410561e-01	 4.1483831e-01	 2.0355575e-01	[ 4.1028122e-01]	 2.1684217e-01


.. parsed-literal::

      21	 4.7799994e-01	 1.9686491e-01	 5.0808148e-01	 1.9499218e-01	[ 5.0463759e-01]	 2.2356725e-01
      22	 5.8787289e-01	 1.9867346e-01	 6.2534338e-01	 1.9401011e-01	[ 6.2454801e-01]	 1.9292831e-01


.. parsed-literal::

      23	 6.2345577e-01	 2.0181673e-01	 6.6032393e-01	 1.9851589e-01	[ 6.6084559e-01]	 1.7737675e-01


.. parsed-literal::

      24	 6.5288185e-01	 1.9485836e-01	 6.8692691e-01	 1.9243622e-01	[ 6.8267155e-01]	 2.1150422e-01


.. parsed-literal::

      25	 6.7048800e-01	 1.9338901e-01	 7.0411434e-01	 1.9080428e-01	[ 6.9469656e-01]	 2.1033072e-01
      26	 6.8915558e-01	 1.9235940e-01	 7.2374874e-01	 1.8943215e-01	[ 7.1081867e-01]	 1.8761611e-01


.. parsed-literal::

      27	 7.1739482e-01	 1.8805234e-01	 7.5290231e-01	 1.8774940e-01	[ 7.3207266e-01]	 2.1118927e-01
      28	 7.4925261e-01	 1.8598823e-01	 7.8589183e-01	 1.8698846e-01	[ 7.5822093e-01]	 1.9947338e-01


.. parsed-literal::

      29	 7.6971354e-01	 1.8476867e-01	 8.0584838e-01	 1.8707299e-01	[ 7.7672979e-01]	 2.2015095e-01
      30	 7.8844311e-01	 1.8410215e-01	 8.2465676e-01	 1.8651917e-01	[ 7.9589357e-01]	 1.7969656e-01


.. parsed-literal::

      31	 8.0331365e-01	 1.8613322e-01	 8.4048663e-01	 1.8771901e-01	[ 8.1473964e-01]	 2.0600057e-01


.. parsed-literal::

      32	 8.2405880e-01	 1.8778995e-01	 8.6184458e-01	 1.8968699e-01	[ 8.3735869e-01]	 2.0833516e-01
      33	 8.3971534e-01	 1.8822686e-01	 8.7734501e-01	 1.8976504e-01	[ 8.5651814e-01]	 1.8799114e-01


.. parsed-literal::

      34	 8.5593146e-01	 1.8721533e-01	 8.9414704e-01	 1.8793637e-01	[ 8.7587723e-01]	 2.0962334e-01


.. parsed-literal::

      35	 8.8017337e-01	 1.8726393e-01	 9.1879212e-01	 1.8495123e-01	[ 9.0939773e-01]	 2.1206212e-01


.. parsed-literal::

      36	 8.9937667e-01	 1.8675876e-01	 9.3865540e-01	 1.8317843e-01	[ 9.3508449e-01]	 2.1870589e-01


.. parsed-literal::

      37	 9.2043545e-01	 1.8799610e-01	 9.6097231e-01	 1.8345919e-01	[ 9.5360634e-01]	 2.1401739e-01


.. parsed-literal::

      38	 9.3374223e-01	 1.8749033e-01	 9.7469007e-01	 1.8290254e-01	[ 9.6206665e-01]	 2.1982789e-01


.. parsed-literal::

      39	 9.4391804e-01	 1.8601109e-01	 9.8464044e-01	 1.8195209e-01	[ 9.7030206e-01]	 2.1005154e-01


.. parsed-literal::

      40	 9.5771531e-01	 1.8277357e-01	 9.9835073e-01	 1.8059882e-01	[ 9.7892894e-01]	 2.0855522e-01


.. parsed-literal::

      41	 9.6812861e-01	 1.8078520e-01	 1.0093813e+00	 1.8024678e-01	[ 9.8125320e-01]	 2.0494652e-01
      42	 9.7866035e-01	 1.7898379e-01	 1.0204658e+00	 1.7931951e-01	[ 9.9060753e-01]	 2.0564651e-01


.. parsed-literal::

      43	 9.8808302e-01	 1.7900205e-01	 1.0306019e+00	 1.7863976e-01	[ 1.0009494e+00]	 2.1131754e-01
      44	 1.0002420e+00	 1.7861585e-01	 1.0437244e+00	 1.7787584e-01	[ 1.0150976e+00]	 1.9244051e-01


.. parsed-literal::

      45	 1.0096116e+00	 1.7899182e-01	 1.0537144e+00	 1.7771027e-01	[ 1.0246453e+00]	 2.0443273e-01


.. parsed-literal::

      46	 1.0189770e+00	 1.7780657e-01	 1.0629661e+00	 1.7743989e-01	[ 1.0355163e+00]	 2.1162176e-01


.. parsed-literal::

      47	 1.0293459e+00	 1.7653178e-01	 1.0735235e+00	 1.7761552e-01	[ 1.0474238e+00]	 2.1999407e-01


.. parsed-literal::

      48	 1.0367065e+00	 1.7642220e-01	 1.0809480e+00	 1.7731109e-01	[ 1.0501633e+00]	 2.1261430e-01


.. parsed-literal::

      49	 1.0450093e+00	 1.7582908e-01	 1.0891224e+00	 1.7639151e-01	[ 1.0584733e+00]	 2.1627951e-01


.. parsed-literal::

      50	 1.0553377e+00	 1.7509138e-01	 1.0997353e+00	 1.7509935e-01	[ 1.0686495e+00]	 2.0657444e-01


.. parsed-literal::

      51	 1.0624136e+00	 1.7349084e-01	 1.1072242e+00	 1.7370116e-01	[ 1.0691052e+00]	 2.0978713e-01


.. parsed-literal::

      52	 1.0688115e+00	 1.7287534e-01	 1.1136590e+00	 1.7350534e-01	[ 1.0720638e+00]	 2.1084571e-01


.. parsed-literal::

      53	 1.0729082e+00	 1.7239982e-01	 1.1177097e+00	 1.7319964e-01	[ 1.0746776e+00]	 2.0799708e-01
      54	 1.0815231e+00	 1.7111505e-01	 1.1266140e+00	 1.7200967e-01	[ 1.0788592e+00]	 1.8305278e-01


.. parsed-literal::

      55	 1.0868999e+00	 1.7117293e-01	 1.1323433e+00	 1.7114554e-01	  1.0734418e+00 	 2.1203589e-01


.. parsed-literal::

      56	 1.0930937e+00	 1.6957593e-01	 1.1383975e+00	 1.7025643e-01	[ 1.0839437e+00]	 2.1488285e-01


.. parsed-literal::

      57	 1.0980214e+00	 1.6825005e-01	 1.1433188e+00	 1.6937764e-01	[ 1.0923747e+00]	 2.1360588e-01


.. parsed-literal::

      58	 1.1030804e+00	 1.6727874e-01	 1.1484539e+00	 1.6855740e-01	[ 1.0988405e+00]	 2.0818400e-01


.. parsed-literal::

      59	 1.1090758e+00	 1.6648568e-01	 1.1549121e+00	 1.6772040e-01	[ 1.1086600e+00]	 2.1465898e-01


.. parsed-literal::

      60	 1.1165737e+00	 1.6540844e-01	 1.1626049e+00	 1.6673717e-01	[ 1.1101823e+00]	 2.0893860e-01


.. parsed-literal::

      61	 1.1216795e+00	 1.6471739e-01	 1.1677991e+00	 1.6625825e-01	[ 1.1109117e+00]	 2.1361947e-01


.. parsed-literal::

      62	 1.1287740e+00	 1.6365611e-01	 1.1752284e+00	 1.6526443e-01	[ 1.1130144e+00]	 2.0775890e-01


.. parsed-literal::

      63	 1.1329521e+00	 1.6267213e-01	 1.1797159e+00	 1.6434565e-01	[ 1.1175184e+00]	 3.2172084e-01


.. parsed-literal::

      64	 1.1376792e+00	 1.6205686e-01	 1.1846351e+00	 1.6346188e-01	[ 1.1204688e+00]	 2.1604204e-01
      65	 1.1422475e+00	 1.6128818e-01	 1.1892576e+00	 1.6270180e-01	[ 1.1256566e+00]	 1.8096232e-01


.. parsed-literal::

      66	 1.1466711e+00	 1.6055461e-01	 1.1937258e+00	 1.6180082e-01	[ 1.1265084e+00]	 2.1711922e-01


.. parsed-literal::

      67	 1.1513500e+00	 1.6009127e-01	 1.1983699e+00	 1.6157496e-01	[ 1.1313796e+00]	 2.1513152e-01


.. parsed-literal::

      68	 1.1562444e+00	 1.5944106e-01	 1.2034585e+00	 1.6112709e-01	[ 1.1340295e+00]	 2.1355677e-01
      69	 1.1623736e+00	 1.5889084e-01	 1.2098184e+00	 1.6057995e-01	[ 1.1366422e+00]	 1.9842362e-01


.. parsed-literal::

      70	 1.1647150e+00	 1.5766706e-01	 1.2126541e+00	 1.5979754e-01	  1.1360312e+00 	 2.1203113e-01


.. parsed-literal::

      71	 1.1710681e+00	 1.5764272e-01	 1.2187457e+00	 1.5976833e-01	[ 1.1414787e+00]	 2.2066903e-01


.. parsed-literal::

      72	 1.1731386e+00	 1.5753810e-01	 1.2206600e+00	 1.5987184e-01	[ 1.1439083e+00]	 2.0401716e-01


.. parsed-literal::

      73	 1.1773113e+00	 1.5682895e-01	 1.2247506e+00	 1.5980821e-01	[ 1.1487736e+00]	 2.1576786e-01
      74	 1.1828953e+00	 1.5576534e-01	 1.2303176e+00	 1.5989326e-01	[ 1.1508647e+00]	 1.7470336e-01


.. parsed-literal::

      75	 1.1844515e+00	 1.5368758e-01	 1.2324049e+00	 1.6004391e-01	[ 1.1545586e+00]	 2.1852303e-01
      76	 1.1913019e+00	 1.5344690e-01	 1.2390933e+00	 1.5952446e-01	[ 1.1546362e+00]	 1.8164802e-01


.. parsed-literal::

      77	 1.1943275e+00	 1.5295493e-01	 1.2422791e+00	 1.5915603e-01	  1.1533083e+00 	 2.1151924e-01


.. parsed-literal::

      78	 1.1989411e+00	 1.5194403e-01	 1.2472702e+00	 1.5844540e-01	  1.1529783e+00 	 2.0695972e-01


.. parsed-literal::

      79	 1.2043694e+00	 1.5101007e-01	 1.2530070e+00	 1.5775462e-01	[ 1.1596186e+00]	 2.0613313e-01


.. parsed-literal::

      80	 1.2087836e+00	 1.5006174e-01	 1.2579520e+00	 1.5717670e-01	[ 1.1645649e+00]	 2.1593618e-01


.. parsed-literal::

      81	 1.2130543e+00	 1.5008799e-01	 1.2620158e+00	 1.5724232e-01	[ 1.1709122e+00]	 2.1044755e-01


.. parsed-literal::

      82	 1.2156073e+00	 1.5016131e-01	 1.2643876e+00	 1.5737343e-01	[ 1.1749372e+00]	 2.0165896e-01


.. parsed-literal::

      83	 1.2197361e+00	 1.4964037e-01	 1.2686015e+00	 1.5750778e-01	[ 1.1763878e+00]	 2.0881915e-01


.. parsed-literal::

      84	 1.2244252e+00	 1.4970249e-01	 1.2734660e+00	 1.5767800e-01	[ 1.1806888e+00]	 2.2055650e-01


.. parsed-literal::

      85	 1.2277923e+00	 1.4920505e-01	 1.2771054e+00	 1.5764263e-01	  1.1791003e+00 	 2.1258855e-01


.. parsed-literal::

      86	 1.2321886e+00	 1.4829931e-01	 1.2819813e+00	 1.5743780e-01	  1.1787858e+00 	 2.0132089e-01


.. parsed-literal::

      87	 1.2358531e+00	 1.4806995e-01	 1.2858764e+00	 1.5737349e-01	  1.1762378e+00 	 2.1272635e-01
      88	 1.2398426e+00	 1.4781091e-01	 1.2898241e+00	 1.5709520e-01	  1.1802251e+00 	 2.0402813e-01


.. parsed-literal::

      89	 1.2444640e+00	 1.4745945e-01	 1.2943444e+00	 1.5654459e-01	[ 1.1876073e+00]	 2.1404529e-01


.. parsed-literal::

      90	 1.2475438e+00	 1.4732185e-01	 1.2973554e+00	 1.5618482e-01	[ 1.1888628e+00]	 2.2122002e-01
      91	 1.2508987e+00	 1.4688612e-01	 1.3007171e+00	 1.5558395e-01	[ 1.1923695e+00]	 1.8900490e-01


.. parsed-literal::

      92	 1.2558332e+00	 1.4661265e-01	 1.3059145e+00	 1.5474749e-01	[ 1.1952441e+00]	 1.9062781e-01
      93	 1.2589318e+00	 1.4610401e-01	 1.3093290e+00	 1.5345068e-01	[ 1.1984152e+00]	 1.7395163e-01


.. parsed-literal::

      94	 1.2629055e+00	 1.4588545e-01	 1.3133963e+00	 1.5336575e-01	[ 1.2016054e+00]	 2.0592904e-01


.. parsed-literal::

      95	 1.2654948e+00	 1.4593293e-01	 1.3159829e+00	 1.5342783e-01	[ 1.2051763e+00]	 2.0241046e-01


.. parsed-literal::

      96	 1.2685295e+00	 1.4556862e-01	 1.3191890e+00	 1.5320031e-01	[ 1.2070858e+00]	 2.1324897e-01


.. parsed-literal::

      97	 1.2728447e+00	 1.4511062e-01	 1.3236873e+00	 1.5303653e-01	[ 1.2080371e+00]	 2.0756078e-01


.. parsed-literal::

      98	 1.2764422e+00	 1.4464773e-01	 1.3272711e+00	 1.5274323e-01	[ 1.2101774e+00]	 2.1219301e-01


.. parsed-literal::

      99	 1.2785317e+00	 1.4453224e-01	 1.3292626e+00	 1.5275037e-01	  1.2096204e+00 	 2.1286297e-01


.. parsed-literal::

     100	 1.2814727e+00	 1.4414388e-01	 1.3322288e+00	 1.5267215e-01	  1.2084518e+00 	 2.2146559e-01


.. parsed-literal::

     101	 1.2838511e+00	 1.4384976e-01	 1.3347862e+00	 1.5260667e-01	  1.2036361e+00 	 2.0824885e-01


.. parsed-literal::

     102	 1.2865919e+00	 1.4362336e-01	 1.3376558e+00	 1.5243008e-01	  1.2066531e+00 	 2.0272446e-01
     103	 1.2900291e+00	 1.4344782e-01	 1.3413161e+00	 1.5205855e-01	  1.2093015e+00 	 1.9905257e-01


.. parsed-literal::

     104	 1.2929678e+00	 1.4355499e-01	 1.3443547e+00	 1.5161559e-01	[ 1.2155262e+00]	 2.0668936e-01


.. parsed-literal::

     105	 1.2960782e+00	 1.4386986e-01	 1.3474347e+00	 1.5134428e-01	[ 1.2174560e+00]	 2.1278834e-01


.. parsed-literal::

     106	 1.2989422e+00	 1.4404873e-01	 1.3502172e+00	 1.5110112e-01	[ 1.2187242e+00]	 2.1429753e-01


.. parsed-literal::

     107	 1.3009270e+00	 1.4390362e-01	 1.3521375e+00	 1.5037236e-01	  1.2180506e+00 	 2.1697497e-01
     108	 1.3031447e+00	 1.4377033e-01	 1.3543281e+00	 1.5028550e-01	[ 1.2202375e+00]	 1.8308663e-01


.. parsed-literal::

     109	 1.3058335e+00	 1.4336473e-01	 1.3571268e+00	 1.4998151e-01	[ 1.2216545e+00]	 2.1622515e-01


.. parsed-literal::

     110	 1.3081588e+00	 1.4286150e-01	 1.3595999e+00	 1.4955461e-01	[ 1.2219725e+00]	 2.1378112e-01


.. parsed-literal::

     111	 1.3119294e+00	 1.4207729e-01	 1.3636173e+00	 1.4894484e-01	[ 1.2239821e+00]	 2.0779324e-01
     112	 1.3145013e+00	 1.4127629e-01	 1.3663280e+00	 1.4845299e-01	  1.2209210e+00 	 1.9896173e-01


.. parsed-literal::

     113	 1.3168858e+00	 1.4134871e-01	 1.3685378e+00	 1.4866509e-01	[ 1.2240913e+00]	 2.1150661e-01
     114	 1.3188760e+00	 1.4127515e-01	 1.3704225e+00	 1.4875091e-01	[ 1.2273347e+00]	 1.7123365e-01


.. parsed-literal::

     115	 1.3214265e+00	 1.4104230e-01	 1.3729773e+00	 1.4873435e-01	[ 1.2287523e+00]	 2.1271515e-01
     116	 1.3235144e+00	 1.4026510e-01	 1.3752328e+00	 1.4807381e-01	[ 1.2362261e+00]	 1.8808818e-01


.. parsed-literal::

     117	 1.3265480e+00	 1.3999576e-01	 1.3782895e+00	 1.4790743e-01	[ 1.2367632e+00]	 2.0974803e-01


.. parsed-literal::

     118	 1.3282016e+00	 1.3973956e-01	 1.3800610e+00	 1.4761862e-01	[ 1.2368050e+00]	 2.1574402e-01


.. parsed-literal::

     119	 1.3306106e+00	 1.3937021e-01	 1.3826321e+00	 1.4717546e-01	[ 1.2389102e+00]	 2.0945215e-01
     120	 1.3340348e+00	 1.3895612e-01	 1.3863944e+00	 1.4665123e-01	  1.2371805e+00 	 1.9066000e-01


.. parsed-literal::

     121	 1.3363946e+00	 1.3851151e-01	 1.3889743e+00	 1.4626440e-01	[ 1.2426007e+00]	 1.8073106e-01


.. parsed-literal::

     122	 1.3383426e+00	 1.3845614e-01	 1.3907834e+00	 1.4638136e-01	[ 1.2430277e+00]	 2.1140409e-01


.. parsed-literal::

     123	 1.3398131e+00	 1.3851872e-01	 1.3921146e+00	 1.4666224e-01	  1.2430116e+00 	 2.1333885e-01
     124	 1.3416815e+00	 1.3826966e-01	 1.3940256e+00	 1.4670044e-01	  1.2423022e+00 	 1.7432308e-01


.. parsed-literal::

     125	 1.3442404e+00	 1.3807113e-01	 1.3966641e+00	 1.4675020e-01	  1.2421128e+00 	 2.1665955e-01


.. parsed-literal::

     126	 1.3469464e+00	 1.3788726e-01	 1.3996218e+00	 1.4666077e-01	  1.2411449e+00 	 2.1884942e-01


.. parsed-literal::

     127	 1.3486582e+00	 1.3731285e-01	 1.4016998e+00	 1.4608024e-01	  1.2405060e+00 	 2.0880413e-01
     128	 1.3510006e+00	 1.3742719e-01	 1.4040062e+00	 1.4603496e-01	  1.2428614e+00 	 1.8202615e-01


.. parsed-literal::

     129	 1.3537173e+00	 1.3751164e-01	 1.4067552e+00	 1.4585153e-01	[ 1.2455302e+00]	 2.0819521e-01


.. parsed-literal::

     130	 1.3557068e+00	 1.3750116e-01	 1.4087732e+00	 1.4569452e-01	[ 1.2472724e+00]	 2.1705985e-01


.. parsed-literal::

     131	 1.3592351e+00	 1.3757429e-01	 1.4123479e+00	 1.4529586e-01	[ 1.2497456e+00]	 2.1489906e-01


.. parsed-literal::

     132	 1.3609346e+00	 1.3716974e-01	 1.4141684e+00	 1.4480552e-01	[ 1.2554933e+00]	 2.1290302e-01


.. parsed-literal::

     133	 1.3631379e+00	 1.3716723e-01	 1.4161670e+00	 1.4490328e-01	[ 1.2568616e+00]	 2.2239852e-01


.. parsed-literal::

     134	 1.3641403e+00	 1.3701041e-01	 1.4171843e+00	 1.4479308e-01	  1.2560460e+00 	 2.1084952e-01


.. parsed-literal::

     135	 1.3660520e+00	 1.3657807e-01	 1.4191144e+00	 1.4434906e-01	[ 1.2579054e+00]	 2.2013116e-01


.. parsed-literal::

     136	 1.3681440e+00	 1.3603584e-01	 1.4213736e+00	 1.4374798e-01	  1.2560708e+00 	 2.0900607e-01


.. parsed-literal::

     137	 1.3701283e+00	 1.3587408e-01	 1.4233401e+00	 1.4344214e-01	[ 1.2586562e+00]	 2.1605682e-01


.. parsed-literal::

     138	 1.3722884e+00	 1.3559746e-01	 1.4254851e+00	 1.4296861e-01	[ 1.2617724e+00]	 2.1926117e-01


.. parsed-literal::

     139	 1.3736043e+00	 1.3575130e-01	 1.4267636e+00	 1.4289576e-01	[ 1.2643709e+00]	 2.2165751e-01
     140	 1.3747016e+00	 1.3574213e-01	 1.4278163e+00	 1.4289908e-01	[ 1.2648795e+00]	 1.9053149e-01


.. parsed-literal::

     141	 1.3775847e+00	 1.3554184e-01	 1.4306666e+00	 1.4270201e-01	  1.2641100e+00 	 2.1092749e-01
     142	 1.3789679e+00	 1.3548862e-01	 1.4320479e+00	 1.4266505e-01	  1.2639405e+00 	 1.7792964e-01


.. parsed-literal::

     143	 1.3807574e+00	 1.3520036e-01	 1.4338523e+00	 1.4242665e-01	  1.2645671e+00 	 2.1139264e-01


.. parsed-literal::

     144	 1.3832044e+00	 1.3481668e-01	 1.4363568e+00	 1.4210511e-01	  1.2637211e+00 	 2.1049619e-01


.. parsed-literal::

     145	 1.3845568e+00	 1.3444363e-01	 1.4377988e+00	 1.4195856e-01	  1.2641340e+00 	 2.0407510e-01


.. parsed-literal::

     146	 1.3861149e+00	 1.3439854e-01	 1.4393814e+00	 1.4198778e-01	  1.2624292e+00 	 2.1083093e-01


.. parsed-literal::

     147	 1.3878735e+00	 1.3439541e-01	 1.4412036e+00	 1.4211136e-01	  1.2590239e+00 	 2.0781946e-01


.. parsed-literal::

     148	 1.3890678e+00	 1.3426774e-01	 1.4424720e+00	 1.4214573e-01	  1.2560224e+00 	 2.0753932e-01


.. parsed-literal::

     149	 1.3904905e+00	 1.3416415e-01	 1.4439548e+00	 1.4218018e-01	  1.2537647e+00 	 2.1344042e-01


.. parsed-literal::

     150	 1.3924631e+00	 1.3384082e-01	 1.4460251e+00	 1.4208694e-01	  1.2509613e+00 	 2.0892072e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.16 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f74ac751360>



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
    CPU times: user 1.89 s, sys: 46 ms, total: 1.94 s
    Wall time: 646 ms


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

