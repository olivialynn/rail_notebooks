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
       1	-3.4023891e-01	 3.1935777e-01	-3.3068471e-01	 3.2537670e-01	[-3.4220314e-01]	 4.6605015e-01


.. parsed-literal::

       2	-2.7218004e-01	 3.1031877e-01	-2.4984054e-01	 3.1471883e-01	[-2.6328978e-01]	 2.3197770e-01


.. parsed-literal::

       3	-2.2727057e-01	 2.8999599e-01	-1.8465877e-01	 2.9153845e-01	[-1.9180822e-01]	 2.8291726e-01


.. parsed-literal::

       4	-1.9451779e-01	 2.7373564e-01	-1.4688971e-01	 2.7562243e-01	[-1.5746291e-01]	 3.2045579e-01
       5	-1.3131577e-01	 2.5728179e-01	-9.2509953e-02	 2.6027178e-01	[-1.0812955e-01]	 2.0318103e-01


.. parsed-literal::

       6	-6.8171152e-02	 2.5096239e-01	-3.7590575e-02	 2.5487332e-01	[-5.3385966e-02]	 1.9228363e-01


.. parsed-literal::

       7	-4.5279744e-02	 2.4714622e-01	-2.1187715e-02	 2.5272712e-01	[-4.3271274e-02]	 2.1830773e-01


.. parsed-literal::

       8	-3.0209554e-02	 2.4456137e-01	-1.1342596e-02	 2.5059604e-01	[-3.6932246e-02]	 2.0385265e-01


.. parsed-literal::

       9	-2.2205314e-02	 2.4314943e-01	-4.8484487e-03	 2.5027069e-01	[-3.2012543e-02]	 2.1027040e-01


.. parsed-literal::

      10	-1.4980313e-02	 2.4181241e-01	 1.0480943e-03	 2.4979930e-01	[-2.9149333e-02]	 2.0548797e-01
      11	-6.3995447e-03	 2.4016674e-01	 8.3959937e-03	 2.4825420e-01	[-2.0909855e-02]	 1.8353057e-01


.. parsed-literal::

      12	-1.4442850e-03	 2.3945337e-01	 1.2597663e-02	 2.4769787e-01	[-1.8768466e-02]	 1.9980502e-01


.. parsed-literal::

      13	 2.0332745e-03	 2.3870440e-01	 1.6121453e-02	 2.4664826e-01	[-1.4435923e-02]	 2.0707989e-01


.. parsed-literal::

      14	 1.6413885e-02	 2.3592667e-01	 3.2014860e-02	 2.4182414e-01	[ 7.4272289e-03]	 2.1413398e-01


.. parsed-literal::

      15	 9.9311649e-02	 2.2630436e-01	 1.2134098e-01	 2.3092938e-01	[ 1.0542481e-01]	 2.1589899e-01
      16	 1.3561785e-01	 2.2393859e-01	 1.5683170e-01	 2.3098126e-01	[ 1.3682245e-01]	 1.9147348e-01


.. parsed-literal::

      17	 2.0388574e-01	 2.1819343e-01	 2.2875352e-01	 2.2774216e-01	[ 1.9980546e-01]	 2.0495415e-01
      18	 2.6915898e-01	 2.1428598e-01	 3.0040842e-01	 2.2256315e-01	[ 2.6464927e-01]	 1.7636347e-01


.. parsed-literal::

      19	 3.0485325e-01	 2.1286509e-01	 3.3531697e-01	 2.1668529e-01	[ 3.1405075e-01]	 2.0288849e-01
      20	 3.6438485e-01	 2.0442983e-01	 3.9629125e-01	 2.1037704e-01	[ 3.7763560e-01]	 1.9605160e-01


.. parsed-literal::

      21	 4.5244591e-01	 1.9922779e-01	 4.8548923e-01	 2.0811780e-01	[ 4.6123167e-01]	 2.0106459e-01


.. parsed-literal::

      22	 5.8060217e-01	 1.9536546e-01	 6.1663780e-01	 2.0153823e-01	[ 6.0103177e-01]	 2.0936370e-01


.. parsed-literal::

      23	 6.2273953e-01	 1.9271822e-01	 6.6066136e-01	 2.0011606e-01	[ 6.4697371e-01]	 3.3436155e-01


.. parsed-literal::

      24	 6.5080559e-01	 1.9011153e-01	 6.8835413e-01	 1.9673996e-01	[ 6.7741968e-01]	 2.1356463e-01


.. parsed-literal::

      25	 6.7218317e-01	 1.9281566e-01	 7.1027398e-01	 1.9727715e-01	[ 7.0056408e-01]	 2.1132588e-01


.. parsed-literal::

      26	 6.8629949e-01	 2.0111845e-01	 7.2176923e-01	 2.0469141e-01	[ 7.1364449e-01]	 2.1033525e-01
      27	 7.1734906e-01	 1.9508117e-01	 7.5340567e-01	 1.9894581e-01	[ 7.4380975e-01]	 1.9957757e-01


.. parsed-literal::

      28	 7.3687304e-01	 1.9475234e-01	 7.7417500e-01	 1.9814856e-01	[ 7.6585235e-01]	 2.0753956e-01


.. parsed-literal::

      29	 7.7265502e-01	 1.9259102e-01	 8.1134488e-01	 1.9594622e-01	[ 8.0436982e-01]	 2.1419263e-01


.. parsed-literal::

      30	 7.9968953e-01	 1.9057535e-01	 8.3883498e-01	 1.9720020e-01	[ 8.1491465e-01]	 2.1670103e-01
      31	 8.2350965e-01	 1.8509218e-01	 8.6236390e-01	 1.9244432e-01	[ 8.3555546e-01]	 1.9317532e-01


.. parsed-literal::

      32	 8.3741409e-01	 1.8814182e-01	 8.7601718e-01	 1.9544924e-01	[ 8.4870689e-01]	 1.7881727e-01


.. parsed-literal::

      33	 8.5651745e-01	 1.8657466e-01	 8.9577677e-01	 1.9349558e-01	[ 8.6511534e-01]	 2.0839429e-01


.. parsed-literal::

      34	 8.7649454e-01	 1.8427512e-01	 9.1735085e-01	 1.9118954e-01	[ 8.8034778e-01]	 2.2054935e-01


.. parsed-literal::

      35	 8.9554487e-01	 1.8432600e-01	 9.3760469e-01	 1.9122342e-01	[ 9.0276380e-01]	 2.0374846e-01


.. parsed-literal::

      36	 9.1187668e-01	 1.8251477e-01	 9.5381651e-01	 1.8861683e-01	[ 9.2693475e-01]	 2.0771861e-01


.. parsed-literal::

      37	 9.2338057e-01	 1.8200788e-01	 9.6508817e-01	 1.8832307e-01	[ 9.4015403e-01]	 2.1830320e-01


.. parsed-literal::

      38	 9.5003117e-01	 1.8169769e-01	 9.9207233e-01	 1.8956651e-01	[ 9.6789745e-01]	 2.0756626e-01


.. parsed-literal::

      39	 9.5970803e-01	 1.8346259e-01	 1.0030398e+00	 1.9304493e-01	[ 9.7878355e-01]	 2.0171380e-01
      40	 9.8015557e-01	 1.8124178e-01	 1.0237077e+00	 1.9098416e-01	[ 9.9493579e-01]	 2.0761371e-01


.. parsed-literal::

      41	 9.9238315e-01	 1.7943679e-01	 1.0364677e+00	 1.8925874e-01	[ 9.9903666e-01]	 2.1787643e-01


.. parsed-literal::

      42	 1.0021324e+00	 1.7775538e-01	 1.0467419e+00	 1.8762649e-01	[ 1.0082469e+00]	 2.0829630e-01


.. parsed-literal::

      43	 1.0131605e+00	 1.7669551e-01	 1.0585746e+00	 1.8736195e-01	[ 1.0115137e+00]	 2.1339774e-01


.. parsed-literal::

      44	 1.0218485e+00	 1.7515312e-01	 1.0674997e+00	 1.8590106e-01	[ 1.0240853e+00]	 2.0147634e-01
      45	 1.0287915e+00	 1.7471642e-01	 1.0740678e+00	 1.8545072e-01	[ 1.0309927e+00]	 1.8538594e-01


.. parsed-literal::

      46	 1.0399887e+00	 1.7415709e-01	 1.0851503e+00	 1.8468208e-01	[ 1.0374067e+00]	 2.1439767e-01


.. parsed-literal::

      47	 1.0487578e+00	 1.7312388e-01	 1.0942490e+00	 1.8345028e-01	[ 1.0424247e+00]	 2.1012330e-01


.. parsed-literal::

      48	 1.0610770e+00	 1.7142315e-01	 1.1073672e+00	 1.8134386e-01	[ 1.0473492e+00]	 2.1174645e-01
      49	 1.0707472e+00	 1.6906827e-01	 1.1174156e+00	 1.7944047e-01	[ 1.0569534e+00]	 1.8533540e-01


.. parsed-literal::

      50	 1.0802409e+00	 1.6704813e-01	 1.1270182e+00	 1.7779242e-01	[ 1.0673993e+00]	 2.1782517e-01


.. parsed-literal::

      51	 1.0858500e+00	 1.6452135e-01	 1.1331808e+00	 1.7689175e-01	[ 1.0792667e+00]	 2.1788692e-01


.. parsed-literal::

      52	 1.0956659e+00	 1.6424734e-01	 1.1425546e+00	 1.7637303e-01	[ 1.0906073e+00]	 2.0842505e-01


.. parsed-literal::

      53	 1.0997291e+00	 1.6430761e-01	 1.1464199e+00	 1.7625502e-01	[ 1.0936629e+00]	 2.0410681e-01
      54	 1.1090277e+00	 1.6379607e-01	 1.1557813e+00	 1.7585105e-01	[ 1.1003761e+00]	 1.8245935e-01


.. parsed-literal::

      55	 1.1177253e+00	 1.6320975e-01	 1.1646362e+00	 1.7582619e-01	[ 1.1084064e+00]	 2.0922828e-01


.. parsed-literal::

      56	 1.1269103e+00	 1.6129911e-01	 1.1745110e+00	 1.7498878e-01	[ 1.1183520e+00]	 2.0302320e-01
      57	 1.1374518e+00	 1.6058343e-01	 1.1853668e+00	 1.7512640e-01	[ 1.1300260e+00]	 1.9946861e-01


.. parsed-literal::

      58	 1.1473217e+00	 1.5975643e-01	 1.1960048e+00	 1.7539082e-01	[ 1.1365198e+00]	 2.1269464e-01


.. parsed-literal::

      59	 1.1540050e+00	 1.5976054e-01	 1.2027603e+00	 1.7560640e-01	[ 1.1433495e+00]	 2.1184516e-01


.. parsed-literal::

      60	 1.1591398e+00	 1.5934731e-01	 1.2078176e+00	 1.7479084e-01	[ 1.1478370e+00]	 2.1367121e-01


.. parsed-literal::

      61	 1.1725894e+00	 1.5808630e-01	 1.2216770e+00	 1.7279606e-01	[ 1.1509398e+00]	 2.1613789e-01
      62	 1.1774585e+00	 1.5666389e-01	 1.2272997e+00	 1.7020628e-01	  1.1489352e+00 	 1.8016887e-01


.. parsed-literal::

      63	 1.1857220e+00	 1.5620014e-01	 1.2353797e+00	 1.7035528e-01	[ 1.1535825e+00]	 1.8989348e-01


.. parsed-literal::

      64	 1.1921955e+00	 1.5551091e-01	 1.2420786e+00	 1.7023834e-01	  1.1530928e+00 	 2.0800257e-01


.. parsed-literal::

      65	 1.1985386e+00	 1.5493831e-01	 1.2485717e+00	 1.6977023e-01	[ 1.1550620e+00]	 2.1943402e-01
      66	 1.2097381e+00	 1.5377585e-01	 1.2600653e+00	 1.6819461e-01	[ 1.1627216e+00]	 2.0209002e-01


.. parsed-literal::

      67	 1.2142754e+00	 1.5365698e-01	 1.2646351e+00	 1.6696551e-01	[ 1.1638472e+00]	 2.1223474e-01


.. parsed-literal::

      68	 1.2252058e+00	 1.5293701e-01	 1.2752415e+00	 1.6629008e-01	[ 1.1775614e+00]	 2.1050787e-01


.. parsed-literal::

      69	 1.2298000e+00	 1.5261572e-01	 1.2797404e+00	 1.6580019e-01	[ 1.1824757e+00]	 2.0975399e-01


.. parsed-literal::

      70	 1.2346442e+00	 1.5199354e-01	 1.2846720e+00	 1.6501840e-01	[ 1.1877044e+00]	 2.1105337e-01
      71	 1.2421233e+00	 1.5129396e-01	 1.2923391e+00	 1.6417752e-01	[ 1.1933907e+00]	 1.9623423e-01


.. parsed-literal::

      72	 1.2489581e+00	 1.5064559e-01	 1.2993044e+00	 1.6339505e-01	[ 1.1979331e+00]	 1.9288921e-01


.. parsed-literal::

      73	 1.2556074e+00	 1.4970460e-01	 1.3061337e+00	 1.6303959e-01	[ 1.1982219e+00]	 2.0822668e-01
      74	 1.2616738e+00	 1.4896543e-01	 1.3122492e+00	 1.6248042e-01	[ 1.1995485e+00]	 1.8006706e-01


.. parsed-literal::

      75	 1.2659178e+00	 1.4863521e-01	 1.3166251e+00	 1.6268802e-01	  1.1988441e+00 	 2.0918894e-01


.. parsed-literal::

      76	 1.2703085e+00	 1.4836811e-01	 1.3210156e+00	 1.6238286e-01	[ 1.2001011e+00]	 2.1648026e-01


.. parsed-literal::

      77	 1.2758942e+00	 1.4806151e-01	 1.3266906e+00	 1.6212069e-01	[ 1.2015060e+00]	 2.0821190e-01


.. parsed-literal::

      78	 1.2830074e+00	 1.4766498e-01	 1.3339906e+00	 1.6186471e-01	[ 1.2019628e+00]	 2.0904732e-01
      79	 1.2887454e+00	 1.4661551e-01	 1.3400576e+00	 1.6144827e-01	  1.2011316e+00 	 2.0107532e-01


.. parsed-literal::

      80	 1.2971605e+00	 1.4600902e-01	 1.3483489e+00	 1.6088398e-01	[ 1.2040750e+00]	 1.8305659e-01
      81	 1.3025320e+00	 1.4508736e-01	 1.3537180e+00	 1.6017069e-01	[ 1.2065680e+00]	 1.8005061e-01


.. parsed-literal::

      82	 1.3078214e+00	 1.4434082e-01	 1.3591107e+00	 1.5965039e-01	  1.2058975e+00 	 2.1468496e-01


.. parsed-literal::

      83	 1.3141360e+00	 1.4301334e-01	 1.3657257e+00	 1.5906832e-01	  1.2027944e+00 	 2.1907425e-01
      84	 1.3194547e+00	 1.4258843e-01	 1.3710530e+00	 1.5861853e-01	  1.2003935e+00 	 1.8658090e-01


.. parsed-literal::

      85	 1.3237713e+00	 1.4224280e-01	 1.3754201e+00	 1.5829314e-01	  1.1978294e+00 	 2.0019174e-01


.. parsed-literal::

      86	 1.3301058e+00	 1.4155165e-01	 1.3818944e+00	 1.5735526e-01	  1.1956760e+00 	 2.2041702e-01


.. parsed-literal::

      87	 1.3349458e+00	 1.4112797e-01	 1.3869621e+00	 1.5737370e-01	  1.1908247e+00 	 3.1810284e-01


.. parsed-literal::

      88	 1.3398754e+00	 1.4050587e-01	 1.3919663e+00	 1.5635975e-01	  1.1888894e+00 	 2.0781636e-01


.. parsed-literal::

      89	 1.3436632e+00	 1.3988277e-01	 1.3958039e+00	 1.5568328e-01	  1.1922326e+00 	 2.1154141e-01
      90	 1.3492541e+00	 1.3920801e-01	 1.4015463e+00	 1.5505395e-01	  1.1829382e+00 	 1.9056344e-01


.. parsed-literal::

      91	 1.3545168e+00	 1.3855313e-01	 1.4068514e+00	 1.5440549e-01	  1.1862848e+00 	 1.8551898e-01


.. parsed-literal::

      92	 1.3594593e+00	 1.3804471e-01	 1.4118468e+00	 1.5391871e-01	  1.1840042e+00 	 2.0853448e-01


.. parsed-literal::

      93	 1.3655128e+00	 1.3715811e-01	 1.4179893e+00	 1.5307718e-01	  1.1768048e+00 	 2.0580125e-01


.. parsed-literal::

      94	 1.3700829e+00	 1.3622631e-01	 1.4225957e+00	 1.5171613e-01	  1.1866372e+00 	 2.2155070e-01
      95	 1.3740768e+00	 1.3594286e-01	 1.4265186e+00	 1.5135946e-01	  1.1892819e+00 	 1.7667842e-01


.. parsed-literal::

      96	 1.3784327e+00	 1.3548772e-01	 1.4308897e+00	 1.5086512e-01	  1.1909412e+00 	 2.0138955e-01


.. parsed-literal::

      97	 1.3828974e+00	 1.3508930e-01	 1.4355099e+00	 1.5050542e-01	  1.1892427e+00 	 2.1276617e-01


.. parsed-literal::

      98	 1.3870311e+00	 1.3446645e-01	 1.4399574e+00	 1.5025746e-01	  1.1888504e+00 	 2.1223664e-01


.. parsed-literal::

      99	 1.3909929e+00	 1.3430267e-01	 1.4438441e+00	 1.5008290e-01	  1.1887436e+00 	 2.1447206e-01
     100	 1.3945495e+00	 1.3421145e-01	 1.4473377e+00	 1.5009637e-01	  1.1948408e+00 	 1.8166780e-01


.. parsed-literal::

     101	 1.3989488e+00	 1.3376075e-01	 1.4517529e+00	 1.4986339e-01	  1.2009356e+00 	 1.9886136e-01


.. parsed-literal::

     102	 1.4030616e+00	 1.3314811e-01	 1.4559891e+00	 1.4965846e-01	[ 1.2200276e+00]	 2.1278787e-01


.. parsed-literal::

     103	 1.4069901e+00	 1.3308343e-01	 1.4598005e+00	 1.4973743e-01	[ 1.2218440e+00]	 2.0202255e-01
     104	 1.4090552e+00	 1.3288209e-01	 1.4618387e+00	 1.4963061e-01	[ 1.2226963e+00]	 1.7274427e-01


.. parsed-literal::

     105	 1.4125491e+00	 1.3260401e-01	 1.4653900e+00	 1.4987111e-01	  1.2180042e+00 	 1.9116950e-01
     106	 1.4155554e+00	 1.3191780e-01	 1.4685854e+00	 1.4969702e-01	  1.2179073e+00 	 1.7411518e-01


.. parsed-literal::

     107	 1.4188614e+00	 1.3175116e-01	 1.4718859e+00	 1.4990139e-01	  1.2169680e+00 	 2.0717955e-01


.. parsed-literal::

     108	 1.4218472e+00	 1.3139581e-01	 1.4749151e+00	 1.4986398e-01	  1.2174394e+00 	 2.1159792e-01


.. parsed-literal::

     109	 1.4252342e+00	 1.3082703e-01	 1.4783478e+00	 1.4951972e-01	  1.2163003e+00 	 2.1464372e-01


.. parsed-literal::

     110	 1.4278268e+00	 1.3033553e-01	 1.4811338e+00	 1.4934107e-01	  1.2181011e+00 	 2.0847201e-01
     111	 1.4317443e+00	 1.3001812e-01	 1.4849855e+00	 1.4875784e-01	  1.2150034e+00 	 1.9888330e-01


.. parsed-literal::

     112	 1.4337039e+00	 1.2999454e-01	 1.4869006e+00	 1.4860986e-01	  1.2164817e+00 	 2.0936275e-01
     113	 1.4374549e+00	 1.2977216e-01	 1.4907843e+00	 1.4822753e-01	  1.2107003e+00 	 1.8845367e-01


.. parsed-literal::

     114	 1.4388857e+00	 1.2956695e-01	 1.4925560e+00	 1.4795486e-01	  1.2201946e+00 	 1.8417883e-01


.. parsed-literal::

     115	 1.4425656e+00	 1.2945883e-01	 1.4961166e+00	 1.4794299e-01	  1.2105491e+00 	 2.0781446e-01
     116	 1.4442879e+00	 1.2928272e-01	 1.4978448e+00	 1.4787220e-01	  1.2056880e+00 	 1.9933462e-01


.. parsed-literal::

     117	 1.4464844e+00	 1.2904611e-01	 1.5001004e+00	 1.4778391e-01	  1.2000219e+00 	 1.9816399e-01
     118	 1.4490176e+00	 1.2861825e-01	 1.5027350e+00	 1.4742451e-01	  1.1907582e+00 	 1.9960570e-01


.. parsed-literal::

     119	 1.4520982e+00	 1.2846412e-01	 1.5058103e+00	 1.4728997e-01	  1.1924137e+00 	 2.1463370e-01
     120	 1.4540640e+00	 1.2837628e-01	 1.5077644e+00	 1.4717892e-01	  1.1912312e+00 	 2.0017648e-01


.. parsed-literal::

     121	 1.4558547e+00	 1.2819481e-01	 1.5095692e+00	 1.4695769e-01	  1.1895600e+00 	 2.1812391e-01


.. parsed-literal::

     122	 1.4585369e+00	 1.2791595e-01	 1.5123747e+00	 1.4687315e-01	  1.1783036e+00 	 2.1315837e-01


.. parsed-literal::

     123	 1.4612590e+00	 1.2753882e-01	 1.5151915e+00	 1.4641833e-01	  1.1803654e+00 	 2.2546840e-01


.. parsed-literal::

     124	 1.4630278e+00	 1.2736402e-01	 1.5169895e+00	 1.4631316e-01	  1.1808745e+00 	 2.2118616e-01


.. parsed-literal::

     125	 1.4655044e+00	 1.2725327e-01	 1.5195339e+00	 1.4643174e-01	  1.1825903e+00 	 2.0372772e-01
     126	 1.4660961e+00	 1.2672417e-01	 1.5202856e+00	 1.4601935e-01	  1.1586480e+00 	 2.0115447e-01


.. parsed-literal::

     127	 1.4687889e+00	 1.2691148e-01	 1.5228341e+00	 1.4625696e-01	  1.1748000e+00 	 1.9829726e-01
     128	 1.4700964e+00	 1.2691904e-01	 1.5241173e+00	 1.4629288e-01	  1.1762594e+00 	 1.9877338e-01


.. parsed-literal::

     129	 1.4716877e+00	 1.2684752e-01	 1.5256947e+00	 1.4623192e-01	  1.1736126e+00 	 1.9999170e-01


.. parsed-literal::

     130	 1.4740622e+00	 1.2677943e-01	 1.5280746e+00	 1.4616714e-01	  1.1736911e+00 	 2.1418691e-01


.. parsed-literal::

     131	 1.4758693e+00	 1.2643509e-01	 1.5300388e+00	 1.4588400e-01	  1.1594371e+00 	 2.0196104e-01


.. parsed-literal::

     132	 1.4780084e+00	 1.2638524e-01	 1.5320923e+00	 1.4578615e-01	  1.1623883e+00 	 2.0966768e-01


.. parsed-literal::

     133	 1.4794013e+00	 1.2629540e-01	 1.5334648e+00	 1.4572034e-01	  1.1639240e+00 	 2.0377398e-01
     134	 1.4812499e+00	 1.2618198e-01	 1.5353459e+00	 1.4574166e-01	  1.1603956e+00 	 1.9131708e-01


.. parsed-literal::

     135	 1.4826994e+00	 1.2604579e-01	 1.5368877e+00	 1.4557527e-01	  1.1650800e+00 	 3.2047343e-01
     136	 1.4846536e+00	 1.2602347e-01	 1.5388721e+00	 1.4576296e-01	  1.1608309e+00 	 1.9788909e-01


.. parsed-literal::

     137	 1.4861447e+00	 1.2608155e-01	 1.5403660e+00	 1.4596641e-01	  1.1587176e+00 	 2.0806909e-01


.. parsed-literal::

     138	 1.4880933e+00	 1.2621405e-01	 1.5422991e+00	 1.4625560e-01	  1.1615402e+00 	 2.1067905e-01


.. parsed-literal::

     139	 1.4896061e+00	 1.2625419e-01	 1.5438883e+00	 1.4648835e-01	  1.1455465e+00 	 2.1713638e-01


.. parsed-literal::

     140	 1.4915370e+00	 1.2635242e-01	 1.5457463e+00	 1.4665320e-01	  1.1590735e+00 	 2.1304035e-01


.. parsed-literal::

     141	 1.4929319e+00	 1.2620776e-01	 1.5471472e+00	 1.4654078e-01	  1.1604255e+00 	 2.0198321e-01


.. parsed-literal::

     142	 1.4944153e+00	 1.2603829e-01	 1.5486827e+00	 1.4648007e-01	  1.1586526e+00 	 2.0944166e-01


.. parsed-literal::

     143	 1.4957560e+00	 1.2535237e-01	 1.5502475e+00	 1.4591884e-01	  1.1318191e+00 	 2.0351648e-01


.. parsed-literal::

     144	 1.4977347e+00	 1.2540477e-01	 1.5522011e+00	 1.4611764e-01	  1.1361768e+00 	 2.0911717e-01


.. parsed-literal::

     145	 1.4990050e+00	 1.2535289e-01	 1.5534910e+00	 1.4616002e-01	  1.1370198e+00 	 2.1351457e-01


.. parsed-literal::

     146	 1.5002054e+00	 1.2523269e-01	 1.5547469e+00	 1.4611938e-01	  1.1329320e+00 	 2.1395516e-01
     147	 1.5020195e+00	 1.2497974e-01	 1.5566806e+00	 1.4589054e-01	  1.1308118e+00 	 1.9841480e-01


.. parsed-literal::

     148	 1.5042572e+00	 1.2483643e-01	 1.5589629e+00	 1.4587669e-01	  1.1156945e+00 	 1.9423199e-01


.. parsed-literal::

     149	 1.5056783e+00	 1.2477671e-01	 1.5603809e+00	 1.4582093e-01	  1.1073184e+00 	 2.1533251e-01
     150	 1.5069996e+00	 1.2474986e-01	 1.5617362e+00	 1.4583588e-01	  1.0997605e+00 	 2.0056820e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.21 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f03bda53cd0>



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
    CPU times: user 1.83 s, sys: 47.9 ms, total: 1.87 s
    Wall time: 624 ms


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

