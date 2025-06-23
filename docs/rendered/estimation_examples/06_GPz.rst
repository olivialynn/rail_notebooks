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
       1	-3.3443458e-01	 3.1806016e-01	-3.2470067e-01	 3.3254050e-01	[-3.5096478e-01]	 4.7031569e-01


.. parsed-literal::

       2	-2.6605820e-01	 3.0811832e-01	-2.4263157e-01	 3.1911877e-01	[-2.7696189e-01]	 2.3436594e-01


.. parsed-literal::

       3	-2.1988200e-01	 2.8635636e-01	-1.7690558e-01	 2.9977835e-01	[-2.2695472e-01]	 2.9789138e-01
       4	-1.9105831e-01	 2.6353678e-01	-1.4924794e-01	 2.7468897e-01	[-2.1046506e-01]	 1.9053626e-01


.. parsed-literal::

       5	-9.7341829e-02	 2.5570296e-01	-6.2469572e-02	 2.6723015e-01	[-1.1157612e-01]	 2.1458149e-01


.. parsed-literal::

       6	-6.4408876e-02	 2.4997646e-01	-3.3491867e-02	 2.5990477e-01	[-6.9732046e-02]	 2.1335244e-01


.. parsed-literal::

       7	-4.5307860e-02	 2.4707601e-01	-2.1343406e-02	 2.5704205e-01	[-6.1043901e-02]	 2.0842028e-01


.. parsed-literal::

       8	-3.3221232e-02	 2.4507829e-01	-1.2947127e-02	 2.5536831e-01	[-5.5246579e-02]	 2.1395421e-01
       9	-1.9578930e-02	 2.4256432e-01	-2.1390843e-03	 2.5307302e-01	[-4.5421720e-02]	 1.8598485e-01


.. parsed-literal::

      10	-8.5610589e-03	 2.4035873e-01	 6.9501419e-03	 2.5061484e-01	[-3.8127189e-02]	 2.0670176e-01
      11	-4.0781528e-03	 2.3966203e-01	 9.9982923e-03	 2.4881221e-01	[-2.4082909e-02]	 1.8732142e-01


.. parsed-literal::

      12	-2.5550369e-04	 2.3916473e-01	 1.3650265e-02	 2.4828466e-01	[-2.2958556e-02]	 2.1829200e-01


.. parsed-literal::

      13	 2.8084927e-03	 2.3852845e-01	 1.6785026e-02	 2.4741250e-01	[-2.0212614e-02]	 2.0981979e-01


.. parsed-literal::

      14	 1.6067522e-01	 2.2339980e-01	 1.8498450e-01	 2.3535075e-01	[ 1.3200780e-01]	 4.3523836e-01


.. parsed-literal::

      15	 1.8149096e-01	 2.2472052e-01	 2.0631529e-01	 2.3644356e-01	[ 1.6926055e-01]	 3.2441115e-01


.. parsed-literal::

      16	 2.8107249e-01	 2.1624599e-01	 3.1065787e-01	 2.2431841e-01	[ 2.7362819e-01]	 2.1690392e-01


.. parsed-literal::

      17	 3.3998283e-01	 2.1198356e-01	 3.7216526e-01	 2.2219542e-01	[ 3.2574829e-01]	 3.3178639e-01
      18	 3.7672898e-01	 2.0831702e-01	 4.0975935e-01	 2.1480998e-01	[ 3.6546551e-01]	 1.8346477e-01


.. parsed-literal::

      19	 4.1541641e-01	 2.0535573e-01	 4.4850505e-01	 2.1069020e-01	[ 4.1161865e-01]	 2.1707082e-01


.. parsed-literal::

      20	 4.6587108e-01	 2.0461260e-01	 4.9962073e-01	 2.1057815e-01	[ 4.6599062e-01]	 2.1736121e-01


.. parsed-literal::

      21	 5.1768971e-01	 2.0671355e-01	 5.5366403e-01	 2.1332004e-01	[ 5.3356859e-01]	 2.1272326e-01


.. parsed-literal::

      22	 5.8192173e-01	 2.0059551e-01	 6.2010467e-01	 2.0732764e-01	[ 6.0163410e-01]	 2.0810771e-01


.. parsed-literal::

      23	 6.2592968e-01	 1.9662321e-01	 6.6441172e-01	 2.0270339e-01	[ 6.3708099e-01]	 2.1752477e-01
      24	 6.5978946e-01	 1.9611647e-01	 6.9793225e-01	 2.0195002e-01	[ 6.6869503e-01]	 1.9778776e-01


.. parsed-literal::

      25	 6.9481519e-01	 1.9690077e-01	 7.3206390e-01	 2.0214205e-01	[ 6.9636080e-01]	 1.8336105e-01


.. parsed-literal::

      26	 7.2756606e-01	 2.0006050e-01	 7.6573758e-01	 2.0688646e-01	[ 7.1705688e-01]	 2.0104265e-01


.. parsed-literal::

      27	 7.4923511e-01	 2.1334524e-01	 7.8796416e-01	 2.2681531e-01	  7.1395449e-01 	 2.0225573e-01


.. parsed-literal::

      28	 7.8780874e-01	 2.1270124e-01	 8.2869081e-01	 2.2674072e-01	[ 7.6100310e-01]	 2.1791196e-01


.. parsed-literal::

      29	 8.1359690e-01	 2.1310906e-01	 8.5412230e-01	 2.2691073e-01	[ 7.8289156e-01]	 2.0649624e-01


.. parsed-literal::

      30	 8.4152423e-01	 2.1279334e-01	 8.8240758e-01	 2.2644324e-01	[ 8.0981555e-01]	 2.1395230e-01


.. parsed-literal::

      31	 8.7262916e-01	 2.0558727e-01	 9.1409035e-01	 2.1753141e-01	[ 8.3685133e-01]	 2.0895147e-01


.. parsed-literal::

      32	 8.9714222e-01	 2.0034061e-01	 9.3955723e-01	 2.1119051e-01	[ 8.5994588e-01]	 2.1931982e-01


.. parsed-literal::

      33	 9.1312145e-01	 1.9702689e-01	 9.5519137e-01	 2.0646861e-01	[ 8.7486923e-01]	 2.2851133e-01
      34	 9.2297318e-01	 1.8833777e-01	 9.6840515e-01	 1.9473720e-01	  8.6802760e-01 	 1.8602705e-01


.. parsed-literal::

      35	 9.5501781e-01	 1.8737073e-01	 9.9905793e-01	 1.9446594e-01	[ 9.0906943e-01]	 2.1879292e-01


.. parsed-literal::

      36	 9.6780505e-01	 1.8640522e-01	 1.0118173e+00	 1.9386986e-01	[ 9.2645338e-01]	 2.1179438e-01
      37	 9.8892614e-01	 1.8342228e-01	 1.0337430e+00	 1.9032851e-01	[ 9.5526357e-01]	 1.9786739e-01


.. parsed-literal::

      38	 1.0074669e+00	 1.8056497e-01	 1.0541901e+00	 1.8771402e-01	[ 9.7590646e-01]	 2.1367049e-01


.. parsed-literal::

      39	 1.0225274e+00	 1.7825019e-01	 1.0700606e+00	 1.8498654e-01	[ 9.9064956e-01]	 2.1886683e-01


.. parsed-literal::

      40	 1.0303494e+00	 1.7766231e-01	 1.0776670e+00	 1.8419500e-01	[ 9.9517705e-01]	 2.0981026e-01


.. parsed-literal::

      41	 1.0420730e+00	 1.7726621e-01	 1.0892295e+00	 1.8402701e-01	[ 1.0023190e+00]	 2.1729255e-01


.. parsed-literal::

      42	 1.0555524e+00	 1.7591418e-01	 1.1030326e+00	 1.8261671e-01	[ 1.0111139e+00]	 2.1229148e-01
      43	 1.0687017e+00	 1.7501870e-01	 1.1161423e+00	 1.8224290e-01	[ 1.0244142e+00]	 1.9176269e-01


.. parsed-literal::

      44	 1.0766811e+00	 1.7391180e-01	 1.1242087e+00	 1.8080636e-01	[ 1.0329404e+00]	 2.1322322e-01
      45	 1.0906797e+00	 1.7182741e-01	 1.1382690e+00	 1.7874365e-01	[ 1.0435177e+00]	 1.9288540e-01


.. parsed-literal::

      46	 1.1059804e+00	 1.6820332e-01	 1.1543071e+00	 1.7470215e-01	[ 1.0497087e+00]	 1.9109964e-01


.. parsed-literal::

      47	 1.1201556e+00	 1.6662589e-01	 1.1682899e+00	 1.7386675e-01	[ 1.0591487e+00]	 2.1174359e-01


.. parsed-literal::

      48	 1.1311067e+00	 1.6499841e-01	 1.1791794e+00	 1.7252080e-01	[ 1.0678113e+00]	 2.1577954e-01
      49	 1.1468601e+00	 1.6170173e-01	 1.1954663e+00	 1.6971438e-01	[ 1.0807909e+00]	 1.9194102e-01


.. parsed-literal::

      50	 1.1603983e+00	 1.5885641e-01	 1.2093115e+00	 1.6664463e-01	[ 1.0967899e+00]	 2.0873427e-01
      51	 1.1713528e+00	 1.5775736e-01	 1.2202638e+00	 1.6593610e-01	[ 1.1061299e+00]	 1.9329548e-01


.. parsed-literal::

      52	 1.1824459e+00	 1.5600749e-01	 1.2317002e+00	 1.6467460e-01	[ 1.1166346e+00]	 2.1171737e-01


.. parsed-literal::

      53	 1.1932019e+00	 1.5493244e-01	 1.2427931e+00	 1.6346108e-01	[ 1.1245726e+00]	 2.0685101e-01
      54	 1.2064887e+00	 1.5159097e-01	 1.2568543e+00	 1.6109940e-01	[ 1.1366756e+00]	 1.9622660e-01


.. parsed-literal::

      55	 1.2223603e+00	 1.5141171e-01	 1.2724561e+00	 1.6053413e-01	[ 1.1481309e+00]	 2.1387005e-01


.. parsed-literal::

      56	 1.2323836e+00	 1.5050861e-01	 1.2824993e+00	 1.5990966e-01	[ 1.1580422e+00]	 2.1164656e-01


.. parsed-literal::

      57	 1.2457335e+00	 1.4977909e-01	 1.2958464e+00	 1.5942789e-01	[ 1.1735345e+00]	 2.1100760e-01


.. parsed-literal::

      58	 1.2492417e+00	 1.4786258e-01	 1.2999300e+00	 1.6059226e-01	[ 1.1763090e+00]	 2.1156168e-01


.. parsed-literal::

      59	 1.2657872e+00	 1.4703924e-01	 1.3160166e+00	 1.5878830e-01	[ 1.1980088e+00]	 2.1586156e-01


.. parsed-literal::

      60	 1.2721094e+00	 1.4677005e-01	 1.3223563e+00	 1.5820305e-01	[ 1.2022730e+00]	 2.0686769e-01


.. parsed-literal::

      61	 1.2825536e+00	 1.4579691e-01	 1.3332744e+00	 1.5765468e-01	[ 1.2087379e+00]	 2.1049857e-01


.. parsed-literal::

      62	 1.2931507e+00	 1.4424413e-01	 1.3443548e+00	 1.5679866e-01	[ 1.2098859e+00]	 2.1398401e-01
      63	 1.3021069e+00	 1.4333146e-01	 1.3535269e+00	 1.5664911e-01	[ 1.2175142e+00]	 2.0668840e-01


.. parsed-literal::

      64	 1.3088007e+00	 1.4272861e-01	 1.3602519e+00	 1.5656436e-01	[ 1.2239630e+00]	 2.2551870e-01


.. parsed-literal::

      65	 1.3178927e+00	 1.4200012e-01	 1.3695030e+00	 1.5616156e-01	[ 1.2302664e+00]	 2.2059584e-01


.. parsed-literal::

      66	 1.3259314e+00	 1.4181548e-01	 1.3774948e+00	 1.5654668e-01	[ 1.2353520e+00]	 2.2306824e-01
      67	 1.3321900e+00	 1.4129789e-01	 1.3837729e+00	 1.5677721e-01	[ 1.2393635e+00]	 1.9632411e-01


.. parsed-literal::

      68	 1.3413314e+00	 1.4103042e-01	 1.3930688e+00	 1.5744260e-01	[ 1.2430507e+00]	 2.1777010e-01


.. parsed-literal::

      69	 1.3479874e+00	 1.3985644e-01	 1.3998979e+00	 1.5739471e-01	[ 1.2436097e+00]	 2.1323037e-01


.. parsed-literal::

      70	 1.3550936e+00	 1.3943786e-01	 1.4069789e+00	 1.5754571e-01	[ 1.2515455e+00]	 2.1418667e-01


.. parsed-literal::

      71	 1.3606173e+00	 1.3908956e-01	 1.4126082e+00	 1.5780640e-01	[ 1.2539248e+00]	 2.1528196e-01


.. parsed-literal::

      72	 1.3673713e+00	 1.3826764e-01	 1.4195330e+00	 1.5816777e-01	[ 1.2549908e+00]	 2.3225927e-01
      73	 1.3751783e+00	 1.3642098e-01	 1.4276377e+00	 1.5893215e-01	  1.2479161e+00 	 1.9062829e-01


.. parsed-literal::

      74	 1.3833883e+00	 1.3534973e-01	 1.4357603e+00	 1.5921211e-01	  1.2545671e+00 	 2.1447802e-01


.. parsed-literal::

      75	 1.3883345e+00	 1.3476974e-01	 1.4406400e+00	 1.5913295e-01	[ 1.2581268e+00]	 2.0536590e-01


.. parsed-literal::

      76	 1.3951903e+00	 1.3366445e-01	 1.4476186e+00	 1.5863012e-01	[ 1.2592223e+00]	 2.1366096e-01
      77	 1.3966477e+00	 1.3295607e-01	 1.4492756e+00	 1.5906550e-01	  1.2550058e+00 	 1.8778777e-01


.. parsed-literal::

      78	 1.4020130e+00	 1.3298432e-01	 1.4544737e+00	 1.5838298e-01	[ 1.2626360e+00]	 2.1676660e-01
      79	 1.4049194e+00	 1.3272699e-01	 1.4574131e+00	 1.5826729e-01	[ 1.2634770e+00]	 1.7394376e-01


.. parsed-literal::

      80	 1.4089428e+00	 1.3240407e-01	 1.4615121e+00	 1.5789729e-01	[ 1.2671761e+00]	 2.1456194e-01
      81	 1.4134547e+00	 1.3162787e-01	 1.4662002e+00	 1.5782788e-01	  1.2657220e+00 	 1.7979932e-01


.. parsed-literal::

      82	 1.4179316e+00	 1.3087198e-01	 1.4708550e+00	 1.5776537e-01	[ 1.2679082e+00]	 2.2810245e-01


.. parsed-literal::

      83	 1.4217604e+00	 1.3052674e-01	 1.4746608e+00	 1.5751826e-01	[ 1.2755599e+00]	 2.1652722e-01
      84	 1.4255524e+00	 1.3010734e-01	 1.4784813e+00	 1.5775576e-01	[ 1.2774148e+00]	 1.9286704e-01


.. parsed-literal::

      85	 1.4308539e+00	 1.2963547e-01	 1.4838711e+00	 1.5828324e-01	[ 1.2811609e+00]	 2.0917892e-01


.. parsed-literal::

      86	 1.4339408e+00	 1.2937249e-01	 1.4870763e+00	 1.5954412e-01	  1.2736540e+00 	 2.1956706e-01


.. parsed-literal::

      87	 1.4390499e+00	 1.2916647e-01	 1.4920338e+00	 1.5920991e-01	  1.2810446e+00 	 2.0979548e-01


.. parsed-literal::

      88	 1.4411263e+00	 1.2907043e-01	 1.4940263e+00	 1.5897696e-01	[ 1.2835277e+00]	 2.1433806e-01


.. parsed-literal::

      89	 1.4453498e+00	 1.2884785e-01	 1.4982657e+00	 1.5928064e-01	  1.2827530e+00 	 2.1947503e-01


.. parsed-literal::

      90	 1.4497819e+00	 1.2836446e-01	 1.5027910e+00	 1.6035881e-01	  1.2767583e+00 	 2.1294904e-01


.. parsed-literal::

      91	 1.4538380e+00	 1.2826608e-01	 1.5068291e+00	 1.6103217e-01	  1.2749094e+00 	 2.2236872e-01


.. parsed-literal::

      92	 1.4564685e+00	 1.2811580e-01	 1.5095060e+00	 1.6148370e-01	  1.2742418e+00 	 2.0625210e-01


.. parsed-literal::

      93	 1.4593671e+00	 1.2791750e-01	 1.5125406e+00	 1.6177028e-01	  1.2724992e+00 	 2.2258925e-01


.. parsed-literal::

      94	 1.4619995e+00	 1.2770942e-01	 1.5153508e+00	 1.6209687e-01	  1.2712843e+00 	 2.1101761e-01


.. parsed-literal::

      95	 1.4651534e+00	 1.2754255e-01	 1.5184857e+00	 1.6155678e-01	  1.2742125e+00 	 2.1331763e-01


.. parsed-literal::

      96	 1.4674254e+00	 1.2742148e-01	 1.5207888e+00	 1.6121254e-01	  1.2759192e+00 	 2.2162747e-01


.. parsed-literal::

      97	 1.4697995e+00	 1.2728458e-01	 1.5231718e+00	 1.6091571e-01	  1.2783637e+00 	 2.0967388e-01


.. parsed-literal::

      98	 1.4720466e+00	 1.2696834e-01	 1.5255266e+00	 1.6078652e-01	  1.2782154e+00 	 2.1563530e-01


.. parsed-literal::

      99	 1.4748445e+00	 1.2686739e-01	 1.5282516e+00	 1.6090501e-01	  1.2794208e+00 	 2.0218635e-01


.. parsed-literal::

     100	 1.4763430e+00	 1.2674266e-01	 1.5297370e+00	 1.6110658e-01	  1.2782154e+00 	 2.1120381e-01


.. parsed-literal::

     101	 1.4790532e+00	 1.2643225e-01	 1.5324790e+00	 1.6124831e-01	  1.2764817e+00 	 2.1261477e-01


.. parsed-literal::

     102	 1.4807011e+00	 1.2623049e-01	 1.5341484e+00	 1.6133394e-01	  1.2722241e+00 	 3.3542657e-01


.. parsed-literal::

     103	 1.4830785e+00	 1.2603400e-01	 1.5365464e+00	 1.6135370e-01	  1.2723846e+00 	 2.1663237e-01
     104	 1.4855384e+00	 1.2595155e-01	 1.5390280e+00	 1.6139498e-01	  1.2742921e+00 	 1.7921615e-01


.. parsed-literal::

     105	 1.4880588e+00	 1.2599302e-01	 1.5415644e+00	 1.6164080e-01	  1.2758586e+00 	 2.0914793e-01
     106	 1.4900964e+00	 1.2601980e-01	 1.5436494e+00	 1.6174835e-01	  1.2812119e+00 	 1.9209003e-01


.. parsed-literal::

     107	 1.4922133e+00	 1.2596815e-01	 1.5457479e+00	 1.6183031e-01	  1.2794821e+00 	 2.1039581e-01
     108	 1.4942780e+00	 1.2577822e-01	 1.5478404e+00	 1.6181450e-01	  1.2774399e+00 	 1.9395065e-01


.. parsed-literal::

     109	 1.4966297e+00	 1.2543166e-01	 1.5503155e+00	 1.6166659e-01	  1.2736773e+00 	 2.1434689e-01


.. parsed-literal::

     110	 1.4980356e+00	 1.2510369e-01	 1.5520301e+00	 1.6127446e-01	  1.2671809e+00 	 2.1912193e-01


.. parsed-literal::

     111	 1.5007921e+00	 1.2498735e-01	 1.5546902e+00	 1.6119539e-01	  1.2679159e+00 	 2.2067928e-01


.. parsed-literal::

     112	 1.5022431e+00	 1.2488655e-01	 1.5561311e+00	 1.6100578e-01	  1.2685367e+00 	 2.0898604e-01


.. parsed-literal::

     113	 1.5041434e+00	 1.2473189e-01	 1.5580592e+00	 1.6071582e-01	  1.2676202e+00 	 2.2644854e-01


.. parsed-literal::

     114	 1.5058635e+00	 1.2438219e-01	 1.5598638e+00	 1.6006318e-01	  1.2712005e+00 	 2.1255708e-01


.. parsed-literal::

     115	 1.5086738e+00	 1.2425179e-01	 1.5626729e+00	 1.5972620e-01	  1.2691546e+00 	 2.1343064e-01
     116	 1.5099317e+00	 1.2418818e-01	 1.5638892e+00	 1.5956511e-01	  1.2699591e+00 	 1.7777777e-01


.. parsed-literal::

     117	 1.5113805e+00	 1.2404613e-01	 1.5653472e+00	 1.5923172e-01	  1.2703537e+00 	 2.0307350e-01


.. parsed-literal::

     118	 1.5135385e+00	 1.2384943e-01	 1.5675352e+00	 1.5877516e-01	  1.2706205e+00 	 2.2186661e-01
     119	 1.5148191e+00	 1.2361285e-01	 1.5689539e+00	 1.5797115e-01	  1.2677524e+00 	 1.9262171e-01


.. parsed-literal::

     120	 1.5172477e+00	 1.2357597e-01	 1.5712900e+00	 1.5798622e-01	  1.2706539e+00 	 2.1507215e-01
     121	 1.5182928e+00	 1.2356537e-01	 1.5723141e+00	 1.5799180e-01	  1.2713204e+00 	 1.7880297e-01


.. parsed-literal::

     122	 1.5199776e+00	 1.2350466e-01	 1.5740028e+00	 1.5781283e-01	  1.2702127e+00 	 2.1091700e-01


.. parsed-literal::

     123	 1.5222217e+00	 1.2327685e-01	 1.5762904e+00	 1.5727670e-01	  1.2634276e+00 	 2.1573448e-01
     124	 1.5243140e+00	 1.2308768e-01	 1.5784326e+00	 1.5654542e-01	  1.2558553e+00 	 1.8852353e-01


.. parsed-literal::

     125	 1.5256914e+00	 1.2296675e-01	 1.5797988e+00	 1.5633431e-01	  1.2530816e+00 	 2.1836305e-01


.. parsed-literal::

     126	 1.5273354e+00	 1.2266775e-01	 1.5815186e+00	 1.5568206e-01	  1.2495757e+00 	 2.2404385e-01


.. parsed-literal::

     127	 1.5286208e+00	 1.2261530e-01	 1.5828548e+00	 1.5549534e-01	  1.2475673e+00 	 2.1417546e-01


.. parsed-literal::

     128	 1.5300703e+00	 1.2252177e-01	 1.5843299e+00	 1.5517347e-01	  1.2455804e+00 	 2.0697999e-01


.. parsed-literal::

     129	 1.5313283e+00	 1.2217357e-01	 1.5856889e+00	 1.5435234e-01	  1.2443058e+00 	 2.1049380e-01


.. parsed-literal::

     130	 1.5327901e+00	 1.2218347e-01	 1.5870704e+00	 1.5439645e-01	  1.2451458e+00 	 2.1396089e-01


.. parsed-literal::

     131	 1.5336521e+00	 1.2213243e-01	 1.5878840e+00	 1.5434975e-01	  1.2458854e+00 	 2.0541191e-01


.. parsed-literal::

     132	 1.5356531e+00	 1.2193104e-01	 1.5898221e+00	 1.5400252e-01	  1.2447206e+00 	 2.1851039e-01


.. parsed-literal::

     133	 1.5375592e+00	 1.2171290e-01	 1.5917385e+00	 1.5374638e-01	  1.2436203e+00 	 2.1804357e-01


.. parsed-literal::

     134	 1.5391527e+00	 1.2153762e-01	 1.5934313e+00	 1.5361881e-01	  1.2370880e+00 	 2.1825218e-01


.. parsed-literal::

     135	 1.5406085e+00	 1.2149557e-01	 1.5949088e+00	 1.5356546e-01	  1.2387467e+00 	 2.0792866e-01


.. parsed-literal::

     136	 1.5415906e+00	 1.2144679e-01	 1.5959386e+00	 1.5361229e-01	  1.2395149e+00 	 2.2271085e-01


.. parsed-literal::

     137	 1.5429924e+00	 1.2132463e-01	 1.5974175e+00	 1.5349536e-01	  1.2380258e+00 	 2.1846724e-01


.. parsed-literal::

     138	 1.5451371e+00	 1.2099363e-01	 1.5996715e+00	 1.5306737e-01	  1.2353013e+00 	 2.1420097e-01


.. parsed-literal::

     139	 1.5467149e+00	 1.2080706e-01	 1.6012316e+00	 1.5291730e-01	  1.2351342e+00 	 2.0881605e-01


.. parsed-literal::

     140	 1.5478191e+00	 1.2072071e-01	 1.6022649e+00	 1.5287334e-01	  1.2350216e+00 	 2.1904588e-01
     141	 1.5488687e+00	 1.2056223e-01	 1.6032727e+00	 1.5283764e-01	  1.2366876e+00 	 1.7582083e-01


.. parsed-literal::

     142	 1.5499923e+00	 1.2053311e-01	 1.6043865e+00	 1.5303411e-01	  1.2330932e+00 	 1.8178153e-01
     143	 1.5511079e+00	 1.2049007e-01	 1.6055453e+00	 1.5308435e-01	  1.2301990e+00 	 1.8598032e-01


.. parsed-literal::

     144	 1.5528358e+00	 1.2038137e-01	 1.6073917e+00	 1.5307625e-01	  1.2225542e+00 	 1.9487596e-01


.. parsed-literal::

     145	 1.5532551e+00	 1.2026494e-01	 1.6079654e+00	 1.5275411e-01	  1.2135870e+00 	 2.2386694e-01


.. parsed-literal::

     146	 1.5545256e+00	 1.2024439e-01	 1.6091371e+00	 1.5269673e-01	  1.2148578e+00 	 2.1346092e-01


.. parsed-literal::

     147	 1.5551484e+00	 1.2018089e-01	 1.6097392e+00	 1.5255722e-01	  1.2136953e+00 	 2.1287107e-01
     148	 1.5561069e+00	 1.2008682e-01	 1.6106958e+00	 1.5230494e-01	  1.2113929e+00 	 1.9875479e-01


.. parsed-literal::

     149	 1.5574623e+00	 1.1996221e-01	 1.6121004e+00	 1.5194226e-01	  1.2003826e+00 	 2.1240234e-01


.. parsed-literal::

     150	 1.5589506e+00	 1.1989112e-01	 1.6135902e+00	 1.5166261e-01	  1.1975672e+00 	 2.1138406e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 8s, sys: 1.14 s, total: 2min 9s
    Wall time: 32.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f10e49cfc70>



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
    CPU times: user 1.91 s, sys: 39 ms, total: 1.95 s
    Wall time: 660 ms


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

