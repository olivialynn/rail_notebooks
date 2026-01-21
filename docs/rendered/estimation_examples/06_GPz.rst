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
       1	-3.4289710e-01	 3.2082031e-01	-3.3329388e-01	 3.1966771e-01	[-3.3085296e-01]	 4.5151973e-01


.. parsed-literal::

       2	-2.7788288e-01	 3.1236139e-01	-2.5610251e-01	 3.0940466e-01	[-2.4695852e-01]	 2.2464204e-01


.. parsed-literal::

       3	-2.3397778e-01	 2.9197098e-01	-1.9201090e-01	 2.8746654e-01	[-1.7226811e-01]	 2.9256678e-01


.. parsed-literal::

       4	-2.0245211e-01	 2.7601790e-01	-1.5596868e-01	 2.7002771e-01	[-1.2328420e-01]	 2.9202223e-01


.. parsed-literal::

       5	-1.3797490e-01	 2.6004177e-01	-1.0035000e-01	 2.5295267e-01	[-6.1909115e-02]	 2.0365453e-01


.. parsed-literal::

       6	-7.6886002e-02	 2.5356790e-01	-4.7303098e-02	 2.4669361e-01	[-1.9790761e-02]	 2.0066881e-01
       7	-5.5951629e-02	 2.5004414e-01	-3.2504296e-02	 2.4546575e-01	[-1.2689261e-02]	 1.8196368e-01


.. parsed-literal::

       8	-4.0859152e-02	 2.4724745e-01	-2.2398314e-02	 2.4192993e-01	[ 7.6560137e-04]	 1.9701481e-01


.. parsed-literal::

       9	-3.2919032e-02	 2.4592081e-01	-1.5979104e-02	 2.4089903e-01	[ 4.6914137e-03]	 2.0995927e-01


.. parsed-literal::

      10	-2.3821003e-02	 2.4410984e-01	-8.2210865e-03	 2.3966830e-01	[ 1.1416365e-02]	 2.0372200e-01
      11	-1.5748464e-02	 2.4284759e-01	-1.7122595e-03	 2.3951484e-01	[ 1.4899089e-02]	 1.9312501e-01


.. parsed-literal::

      12	-1.0295507e-02	 2.4173107e-01	 3.5063562e-03	 2.3851446e-01	[ 2.0490400e-02]	 2.0635319e-01


.. parsed-literal::

      13	-7.6447547e-03	 2.4123750e-01	 6.0651989e-03	 2.3800831e-01	[ 2.2874965e-02]	 2.0163345e-01
      14	 9.3028915e-03	 2.3809841e-01	 2.4400940e-02	 2.3401920e-01	[ 4.1479538e-02]	 2.0618987e-01


.. parsed-literal::

      15	 9.9800214e-02	 2.2520789e-01	 1.1968434e-01	 2.2446675e-01	[ 1.1829632e-01]	 3.2548904e-01


.. parsed-literal::

      16	 1.3429063e-01	 2.2153845e-01	 1.5598721e-01	 2.2202632e-01	[ 1.5455304e-01]	 2.0711708e-01
      17	 2.2618357e-01	 2.1607903e-01	 2.5238555e-01	 2.1728130e-01	[ 2.5291006e-01]	 2.0177007e-01


.. parsed-literal::

      18	 3.0214080e-01	 2.1078654e-01	 3.3304784e-01	 2.1287742e-01	[ 3.3970017e-01]	 1.8564272e-01


.. parsed-literal::

      19	 3.3664121e-01	 2.0733644e-01	 3.7052710e-01	 2.1100896e-01	[ 3.6238731e-01]	 2.0328021e-01
      20	 3.8141905e-01	 2.0169634e-01	 4.1310706e-01	 2.0628114e-01	[ 4.0140243e-01]	 1.9340467e-01


.. parsed-literal::

      21	 4.2041755e-01	 1.9777651e-01	 4.5367139e-01	 2.0160608e-01	[ 4.3930900e-01]	 1.9923830e-01


.. parsed-literal::

      22	 4.9411510e-01	 1.9437492e-01	 5.2702126e-01	 1.9898615e-01	[ 5.0006189e-01]	 2.1779037e-01
      23	 6.0361550e-01	 1.9176995e-01	 6.3843466e-01	 1.9507866e-01	[ 5.9384711e-01]	 1.8369532e-01


.. parsed-literal::

      24	 6.1610595e-01	 2.0023671e-01	 6.5678085e-01	 1.9944992e-01	  5.7123708e-01 	 2.0016074e-01


.. parsed-literal::

      25	 6.7439966e-01	 1.9268251e-01	 7.1158539e-01	 1.9310141e-01	[ 6.4616474e-01]	 2.1687055e-01
      26	 6.8945283e-01	 1.9131436e-01	 7.2573425e-01	 1.9281692e-01	[ 6.6138699e-01]	 1.7972970e-01


.. parsed-literal::

      27	 7.0541605e-01	 1.8969050e-01	 7.4202858e-01	 1.9131208e-01	[ 6.8015827e-01]	 1.8173170e-01


.. parsed-literal::

      28	 7.2245413e-01	 1.8774139e-01	 7.5926325e-01	 1.9121967e-01	[ 6.8893476e-01]	 2.2122169e-01
      29	 7.3900232e-01	 1.8542877e-01	 7.7612839e-01	 1.9039217e-01	[ 7.0302067e-01]	 1.9803190e-01


.. parsed-literal::

      30	 7.7022164e-01	 1.8491655e-01	 8.0786642e-01	 1.9196197e-01	[ 7.2773066e-01]	 1.7788529e-01


.. parsed-literal::

      31	 7.9750255e-01	 1.8182204e-01	 8.3376927e-01	 1.9074436e-01	[ 7.6491431e-01]	 2.1192002e-01
      32	 8.2151712e-01	 1.8206877e-01	 8.5854915e-01	 1.9141634e-01	[ 7.8467906e-01]	 1.8209076e-01


.. parsed-literal::

      33	 8.4314646e-01	 1.8091601e-01	 8.8141156e-01	 1.8920201e-01	[ 7.9999374e-01]	 2.0634627e-01


.. parsed-literal::

      34	 8.6963593e-01	 1.8010492e-01	 9.0955325e-01	 1.8756101e-01	[ 8.2476258e-01]	 2.0245123e-01


.. parsed-literal::

      35	 8.7882129e-01	 1.7944661e-01	 9.1854559e-01	 1.8676163e-01	[ 8.3763648e-01]	 2.0848250e-01
      36	 8.9150099e-01	 1.7812504e-01	 9.3069172e-01	 1.8621980e-01	[ 8.4962259e-01]	 1.8373513e-01


.. parsed-literal::

      37	 9.0099033e-01	 1.7749847e-01	 9.4018846e-01	 1.8661123e-01	[ 8.5730371e-01]	 2.1039844e-01


.. parsed-literal::

      38	 9.1358839e-01	 1.7680596e-01	 9.5303689e-01	 1.8650324e-01	[ 8.6867663e-01]	 2.1646833e-01


.. parsed-literal::

      39	 9.4153881e-01	 1.7563650e-01	 9.8271239e-01	 1.8960015e-01	[ 8.8416089e-01]	 2.1303844e-01


.. parsed-literal::

      40	 9.5770814e-01	 1.7467961e-01	 9.9896409e-01	 1.8796877e-01	[ 9.1010937e-01]	 2.1405697e-01


.. parsed-literal::

      41	 9.6802236e-01	 1.7348060e-01	 1.0094542e+00	 1.8722855e-01	[ 9.2001921e-01]	 2.0969820e-01
      42	 9.8045429e-01	 1.7269871e-01	 1.0223480e+00	 1.8688087e-01	[ 9.3003977e-01]	 1.8727541e-01


.. parsed-literal::

      43	 9.9071747e-01	 1.7353202e-01	 1.0332796e+00	 1.8788062e-01	[ 9.4602433e-01]	 1.8160987e-01
      44	 1.0002776e+00	 1.7344671e-01	 1.0430224e+00	 1.8743514e-01	[ 9.5065853e-01]	 1.8770003e-01


.. parsed-literal::

      45	 1.0062640e+00	 1.7248151e-01	 1.0488075e+00	 1.8609468e-01	[ 9.5541999e-01]	 1.9889283e-01


.. parsed-literal::

      46	 1.0184210e+00	 1.7179945e-01	 1.0613775e+00	 1.8473420e-01	[ 9.6580337e-01]	 2.1736550e-01
      47	 1.0305127e+00	 1.7066122e-01	 1.0750387e+00	 1.8245398e-01	[ 9.6865492e-01]	 1.8342257e-01


.. parsed-literal::

      48	 1.0389868e+00	 1.7098191e-01	 1.0837485e+00	 1.8131756e-01	[ 9.8754476e-01]	 1.7845607e-01
      49	 1.0446395e+00	 1.6990821e-01	 1.0892144e+00	 1.8047663e-01	[ 9.9287239e-01]	 1.8431568e-01


.. parsed-literal::

      50	 1.0518281e+00	 1.6825354e-01	 1.0965178e+00	 1.7901487e-01	[ 9.9807095e-01]	 1.7991138e-01
      51	 1.0583292e+00	 1.6766954e-01	 1.1031911e+00	 1.7852596e-01	[ 1.0021012e+00]	 1.9802499e-01


.. parsed-literal::

      52	 1.0681231e+00	 1.6657348e-01	 1.1137148e+00	 1.7620827e-01	[ 1.0140559e+00]	 2.0568633e-01
      53	 1.0756377e+00	 1.6596688e-01	 1.1213879e+00	 1.7567885e-01	[ 1.0158979e+00]	 1.8314838e-01


.. parsed-literal::

      54	 1.0801604e+00	 1.6527147e-01	 1.1257174e+00	 1.7557660e-01	[ 1.0182108e+00]	 2.0055532e-01
      55	 1.0855731e+00	 1.6472148e-01	 1.1310835e+00	 1.7464226e-01	[ 1.0245863e+00]	 1.8855977e-01


.. parsed-literal::

      56	 1.0905659e+00	 1.6424314e-01	 1.1362394e+00	 1.7464323e-01	[ 1.0264064e+00]	 2.1682668e-01


.. parsed-literal::

      57	 1.0963014e+00	 1.6362603e-01	 1.1418863e+00	 1.7438849e-01	[ 1.0341265e+00]	 2.0875216e-01
      58	 1.1049027e+00	 1.6243418e-01	 1.1506752e+00	 1.7167575e-01	[ 1.0452957e+00]	 1.7102933e-01


.. parsed-literal::

      59	 1.1110604e+00	 1.6183759e-01	 1.1570448e+00	 1.7013727e-01	[ 1.0474837e+00]	 2.1004963e-01
      60	 1.1160492e+00	 1.6133660e-01	 1.1619987e+00	 1.6922283e-01	[ 1.0507565e+00]	 1.9715214e-01


.. parsed-literal::

      61	 1.1234695e+00	 1.6105095e-01	 1.1696958e+00	 1.6756654e-01	[ 1.0523130e+00]	 2.1358466e-01


.. parsed-literal::

      62	 1.1280877e+00	 1.6093253e-01	 1.1744000e+00	 1.6702273e-01	[ 1.0547650e+00]	 2.0908880e-01


.. parsed-literal::

      63	 1.1357262e+00	 1.6050765e-01	 1.1822561e+00	 1.6685262e-01	[ 1.0578060e+00]	 2.0247650e-01


.. parsed-literal::

      64	 1.1435893e+00	 1.5989456e-01	 1.1904404e+00	 1.6814583e-01	  1.0531522e+00 	 2.1849012e-01
      65	 1.1500519e+00	 1.5850157e-01	 1.1969435e+00	 1.6878655e-01	  1.0574585e+00 	 1.8759012e-01


.. parsed-literal::

      66	 1.1542863e+00	 1.5817892e-01	 1.2010282e+00	 1.6896763e-01	[ 1.0625696e+00]	 1.8266821e-01
      67	 1.1599389e+00	 1.5814372e-01	 1.2067801e+00	 1.6985781e-01	[ 1.0675193e+00]	 2.0129967e-01


.. parsed-literal::

      68	 1.1651462e+00	 1.5771966e-01	 1.2123288e+00	 1.7008952e-01	[ 1.0756313e+00]	 2.0040941e-01


.. parsed-literal::

      69	 1.1704654e+00	 1.5754592e-01	 1.2177358e+00	 1.7096613e-01	[ 1.0770016e+00]	 2.0597196e-01
      70	 1.1738090e+00	 1.5707915e-01	 1.2210267e+00	 1.7081735e-01	[ 1.0811110e+00]	 1.8280029e-01


.. parsed-literal::

      71	 1.1791862e+00	 1.5598082e-01	 1.2264462e+00	 1.7043420e-01	[ 1.0843952e+00]	 2.0825934e-01
      72	 1.1848893e+00	 1.5513622e-01	 1.2323066e+00	 1.6925343e-01	[ 1.0964145e+00]	 1.9444394e-01


.. parsed-literal::

      73	 1.1910547e+00	 1.5427079e-01	 1.2386113e+00	 1.6865062e-01	[ 1.0986483e+00]	 2.0273018e-01
      74	 1.1974857e+00	 1.5328603e-01	 1.2452877e+00	 1.6729717e-01	[ 1.1006041e+00]	 1.9588733e-01


.. parsed-literal::

      75	 1.2015617e+00	 1.5276696e-01	 1.2495162e+00	 1.6652424e-01	[ 1.1061251e+00]	 1.7928743e-01


.. parsed-literal::

      76	 1.2064519e+00	 1.5219490e-01	 1.2544594e+00	 1.6519476e-01	[ 1.1109852e+00]	 2.0555973e-01


.. parsed-literal::

      77	 1.2119250e+00	 1.5144663e-01	 1.2600333e+00	 1.6363497e-01	[ 1.1184260e+00]	 2.0460296e-01
      78	 1.2165155e+00	 1.5070590e-01	 1.2648060e+00	 1.6276444e-01	[ 1.1233832e+00]	 1.7005348e-01


.. parsed-literal::

      79	 1.2213815e+00	 1.4973637e-01	 1.2701276e+00	 1.6228612e-01	[ 1.1277031e+00]	 2.0904732e-01
      80	 1.2276058e+00	 1.4888091e-01	 1.2764149e+00	 1.6166745e-01	[ 1.1304140e+00]	 1.7970729e-01


.. parsed-literal::

      81	 1.2302620e+00	 1.4864242e-01	 1.2789851e+00	 1.6155411e-01	[ 1.1316727e+00]	 2.0871782e-01


.. parsed-literal::

      82	 1.2358125e+00	 1.4782861e-01	 1.2848098e+00	 1.6108380e-01	[ 1.1326765e+00]	 2.1084213e-01
      83	 1.2412129e+00	 1.4675107e-01	 1.2904466e+00	 1.5980524e-01	[ 1.1335674e+00]	 1.7525458e-01


.. parsed-literal::

      84	 1.2460359e+00	 1.4621887e-01	 1.2953281e+00	 1.5908300e-01	[ 1.1406039e+00]	 1.7596865e-01
      85	 1.2510436e+00	 1.4566989e-01	 1.3004101e+00	 1.5827548e-01	[ 1.1451653e+00]	 1.8359423e-01


.. parsed-literal::

      86	 1.2550861e+00	 1.4532936e-01	 1.3045416e+00	 1.5724199e-01	[ 1.1488183e+00]	 1.8156624e-01
      87	 1.2595554e+00	 1.4506116e-01	 1.3090310e+00	 1.5699469e-01	[ 1.1510835e+00]	 2.0558739e-01


.. parsed-literal::

      88	 1.2661300e+00	 1.4433010e-01	 1.3158782e+00	 1.5579514e-01	  1.1468421e+00 	 2.1468163e-01
      89	 1.2708001e+00	 1.4407392e-01	 1.3206375e+00	 1.5567320e-01	  1.1502887e+00 	 1.9584703e-01


.. parsed-literal::

      90	 1.2749396e+00	 1.4374291e-01	 1.3248101e+00	 1.5524334e-01	  1.1502246e+00 	 2.1369648e-01


.. parsed-literal::

      91	 1.2784238e+00	 1.4343173e-01	 1.3283716e+00	 1.5464390e-01	  1.1509108e+00 	 2.1563601e-01


.. parsed-literal::

      92	 1.2815247e+00	 1.4320142e-01	 1.3314769e+00	 1.5451499e-01	[ 1.1526278e+00]	 2.0319796e-01


.. parsed-literal::

      93	 1.2867778e+00	 1.4235491e-01	 1.3370425e+00	 1.5397044e-01	[ 1.1536083e+00]	 2.0753837e-01


.. parsed-literal::

      94	 1.2908858e+00	 1.4223984e-01	 1.3412833e+00	 1.5446241e-01	  1.1430716e+00 	 2.0446181e-01


.. parsed-literal::

      95	 1.2936827e+00	 1.4213152e-01	 1.3439502e+00	 1.5463968e-01	  1.1484538e+00 	 2.0356297e-01


.. parsed-literal::

      96	 1.2971730e+00	 1.4186236e-01	 1.3474458e+00	 1.5433813e-01	  1.1522286e+00 	 2.0150280e-01
      97	 1.2999158e+00	 1.4152876e-01	 1.3503888e+00	 1.5458865e-01	  1.1470432e+00 	 2.0089197e-01


.. parsed-literal::

      98	 1.3031870e+00	 1.4145014e-01	 1.3537100e+00	 1.5447524e-01	  1.1479573e+00 	 2.0611429e-01


.. parsed-literal::

      99	 1.3064485e+00	 1.4130404e-01	 1.3571371e+00	 1.5434621e-01	  1.1462182e+00 	 2.0820880e-01
     100	 1.3090678e+00	 1.4117456e-01	 1.3598327e+00	 1.5425952e-01	  1.1455862e+00 	 1.7873812e-01


.. parsed-literal::

     101	 1.3142640e+00	 1.4068860e-01	 1.3654811e+00	 1.5346640e-01	  1.1409053e+00 	 2.1492171e-01


.. parsed-literal::

     102	 1.3172787e+00	 1.4064359e-01	 1.3685866e+00	 1.5353723e-01	  1.1432147e+00 	 3.1834483e-01
     103	 1.3193307e+00	 1.4059495e-01	 1.3705565e+00	 1.5355325e-01	  1.1452684e+00 	 1.8196082e-01


.. parsed-literal::

     104	 1.3220521e+00	 1.4051864e-01	 1.3733391e+00	 1.5349201e-01	  1.1461531e+00 	 2.1729088e-01


.. parsed-literal::

     105	 1.3247193e+00	 1.4038169e-01	 1.3761338e+00	 1.5361553e-01	  1.1472764e+00 	 2.1354008e-01


.. parsed-literal::

     106	 1.3276752e+00	 1.4031192e-01	 1.3792069e+00	 1.5334915e-01	  1.1487897e+00 	 2.0429635e-01
     107	 1.3304272e+00	 1.4011164e-01	 1.3820787e+00	 1.5299695e-01	  1.1478188e+00 	 1.9940710e-01


.. parsed-literal::

     108	 1.3336565e+00	 1.3972257e-01	 1.3854574e+00	 1.5236388e-01	  1.1461692e+00 	 2.1429420e-01
     109	 1.3367683e+00	 1.3902295e-01	 1.3887970e+00	 1.5137305e-01	  1.1365319e+00 	 1.9245672e-01


.. parsed-literal::

     110	 1.3397383e+00	 1.3864967e-01	 1.3917412e+00	 1.5078340e-01	  1.1393357e+00 	 2.1632409e-01
     111	 1.3427001e+00	 1.3831245e-01	 1.3946637e+00	 1.5036176e-01	  1.1422852e+00 	 1.8485212e-01


.. parsed-literal::

     112	 1.3458300e+00	 1.3795458e-01	 1.3978472e+00	 1.5024937e-01	  1.1444522e+00 	 2.1463776e-01


.. parsed-literal::

     113	 1.3476360e+00	 1.3759859e-01	 1.3999739e+00	 1.5021069e-01	  1.1412685e+00 	 2.0633006e-01


.. parsed-literal::

     114	 1.3511169e+00	 1.3761371e-01	 1.4032940e+00	 1.5064954e-01	  1.1469092e+00 	 2.0573783e-01


.. parsed-literal::

     115	 1.3526678e+00	 1.3761850e-01	 1.4048808e+00	 1.5086669e-01	  1.1475008e+00 	 2.0378184e-01


.. parsed-literal::

     116	 1.3547389e+00	 1.3750599e-01	 1.4070596e+00	 1.5101264e-01	  1.1500255e+00 	 2.0846438e-01


.. parsed-literal::

     117	 1.3570382e+00	 1.3733036e-01	 1.4094968e+00	 1.5101553e-01	  1.1494999e+00 	 2.1004796e-01


.. parsed-literal::

     118	 1.3590584e+00	 1.3715174e-01	 1.4115452e+00	 1.5072679e-01	  1.1506734e+00 	 2.0667696e-01
     119	 1.3624693e+00	 1.3673493e-01	 1.4151376e+00	 1.5016718e-01	  1.1452944e+00 	 1.9533157e-01


.. parsed-literal::

     120	 1.3640232e+00	 1.3641130e-01	 1.4168731e+00	 1.4937223e-01	  1.1451219e+00 	 2.0372629e-01


.. parsed-literal::

     121	 1.3675402e+00	 1.3620086e-01	 1.4203810e+00	 1.4912454e-01	  1.1421086e+00 	 2.2071195e-01


.. parsed-literal::

     122	 1.3696978e+00	 1.3612823e-01	 1.4225348e+00	 1.4897864e-01	  1.1414705e+00 	 2.1310973e-01


.. parsed-literal::

     123	 1.3719619e+00	 1.3601465e-01	 1.4248601e+00	 1.4872919e-01	  1.1398612e+00 	 2.1298552e-01
     124	 1.3739338e+00	 1.3596242e-01	 1.4269838e+00	 1.4823123e-01	  1.1404348e+00 	 1.9883466e-01


.. parsed-literal::

     125	 1.3768528e+00	 1.3577537e-01	 1.4298888e+00	 1.4787721e-01	  1.1401136e+00 	 2.0216084e-01
     126	 1.3792292e+00	 1.3553475e-01	 1.4322862e+00	 1.4741978e-01	  1.1398006e+00 	 1.9689655e-01


.. parsed-literal::

     127	 1.3816502e+00	 1.3524119e-01	 1.4347705e+00	 1.4683606e-01	  1.1398074e+00 	 1.8388391e-01


.. parsed-literal::

     128	 1.3845080e+00	 1.3500107e-01	 1.4377677e+00	 1.4628871e-01	  1.1430223e+00 	 2.0699120e-01


.. parsed-literal::

     129	 1.3876126e+00	 1.3466097e-01	 1.4409381e+00	 1.4552694e-01	  1.1436399e+00 	 2.1823359e-01


.. parsed-literal::

     130	 1.3896603e+00	 1.3462023e-01	 1.4429057e+00	 1.4557900e-01	  1.1490610e+00 	 2.0923400e-01


.. parsed-literal::

     131	 1.3919239e+00	 1.3458729e-01	 1.4451847e+00	 1.4550111e-01	  1.1529324e+00 	 2.0844626e-01
     132	 1.3945820e+00	 1.3447932e-01	 1.4480030e+00	 1.4529102e-01	  1.1522607e+00 	 1.8974805e-01


.. parsed-literal::

     133	 1.3971507e+00	 1.3458929e-01	 1.4506220e+00	 1.4528965e-01	  1.1520874e+00 	 1.8545079e-01
     134	 1.3992347e+00	 1.3457622e-01	 1.4527437e+00	 1.4517744e-01	  1.1498727e+00 	 2.0014477e-01


.. parsed-literal::

     135	 1.4015989e+00	 1.3453310e-01	 1.4551746e+00	 1.4477967e-01	  1.1485885e+00 	 2.0688796e-01
     136	 1.4036482e+00	 1.3453199e-01	 1.4573044e+00	 1.4491312e-01	  1.1519084e+00 	 2.1120715e-01


.. parsed-literal::

     137	 1.4060959e+00	 1.3441476e-01	 1.4597279e+00	 1.4446916e-01	[ 1.1545476e+00]	 2.0390272e-01


.. parsed-literal::

     138	 1.4079185e+00	 1.3432924e-01	 1.4615794e+00	 1.4422484e-01	[ 1.1588550e+00]	 2.1523333e-01
     139	 1.4092613e+00	 1.3423062e-01	 1.4629604e+00	 1.4425696e-01	[ 1.1592535e+00]	 2.0012069e-01


.. parsed-literal::

     140	 1.4112700e+00	 1.3416358e-01	 1.4650755e+00	 1.4448649e-01	[ 1.1599739e+00]	 2.1282983e-01
     141	 1.4130271e+00	 1.3409736e-01	 1.4668294e+00	 1.4471724e-01	  1.1564986e+00 	 1.7162371e-01


.. parsed-literal::

     142	 1.4145357e+00	 1.3407533e-01	 1.4683039e+00	 1.4502846e-01	  1.1542661e+00 	 2.0539117e-01


.. parsed-literal::

     143	 1.4163556e+00	 1.3397257e-01	 1.4701575e+00	 1.4513559e-01	  1.1525509e+00 	 2.0674014e-01
     144	 1.4170171e+00	 1.3382031e-01	 1.4710487e+00	 1.4606481e-01	  1.1522818e+00 	 1.7529774e-01


.. parsed-literal::

     145	 1.4199603e+00	 1.3374043e-01	 1.4738759e+00	 1.4557298e-01	  1.1549273e+00 	 2.1393633e-01
     146	 1.4208239e+00	 1.3370107e-01	 1.4747235e+00	 1.4548144e-01	  1.1565748e+00 	 1.8172860e-01


.. parsed-literal::

     147	 1.4227197e+00	 1.3360642e-01	 1.4766330e+00	 1.4563452e-01	  1.1577826e+00 	 2.0889211e-01
     148	 1.4253726e+00	 1.3348319e-01	 1.4793067e+00	 1.4606353e-01	  1.1546414e+00 	 1.7611122e-01


.. parsed-literal::

     149	 1.4269516e+00	 1.3350282e-01	 1.4809533e+00	 1.4681570e-01	  1.1527593e+00 	 2.8796148e-01


.. parsed-literal::

     150	 1.4293246e+00	 1.3339154e-01	 1.4833223e+00	 1.4722320e-01	  1.1479547e+00 	 2.0084786e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 2s, sys: 1.09 s, total: 2min 3s
    Wall time: 31 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7ff484d14750>



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
    CPU times: user 1.84 s, sys: 47 ms, total: 1.89 s
    Wall time: 610 ms


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




.. image:: 06_GPz_files/06_GPz_16_1.png


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




.. image:: 06_GPz_files/06_GPz_19_1.png

