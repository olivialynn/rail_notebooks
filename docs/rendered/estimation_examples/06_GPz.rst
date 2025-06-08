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
       1	-3.4820514e-01	 3.2165609e-01	-3.3853011e-01	 3.1578845e-01	[-3.2779621e-01]	 4.5818210e-01


.. parsed-literal::

       2	-2.7531237e-01	 3.1058974e-01	-2.5087766e-01	 3.0456165e-01	[-2.3290470e-01]	 2.2916460e-01


.. parsed-literal::

       3	-2.2885099e-01	 2.8871257e-01	-1.8490514e-01	 2.8302373e-01	[-1.6275072e-01]	 2.8225088e-01
       4	-2.0409181e-01	 2.6530723e-01	-1.6393787e-01	 2.6155258e-01	[-1.3790191e-01]	 1.9743347e-01


.. parsed-literal::

       5	-1.0783007e-01	 2.5834558e-01	-7.2092430e-02	 2.5224294e-01	[-4.5035945e-02]	 2.0421553e-01
       6	-7.7287708e-02	 2.5284377e-01	-4.5012927e-02	 2.4821213e-01	[-2.7832605e-02]	 1.9417191e-01


.. parsed-literal::

       7	-5.7598288e-02	 2.4986902e-01	-3.2768190e-02	 2.4501483e-01	[-1.3395548e-02]	 1.9851494e-01


.. parsed-literal::

       8	-4.5995961e-02	 2.4803515e-01	-2.5029378e-02	 2.4316061e-01	[-5.0955534e-03]	 2.0753336e-01


.. parsed-literal::

       9	-3.0941803e-02	 2.4522108e-01	-1.3166366e-02	 2.4067611e-01	[ 5.2676777e-03]	 2.1577930e-01
      10	-2.0000582e-02	 2.4297168e-01	-4.5490393e-03	 2.4047370e-01	[ 7.0601462e-03]	 1.9220352e-01


.. parsed-literal::

      11	-1.3960045e-02	 2.4217661e-01	 3.5742033e-04	 2.3870814e-01	[ 1.4750286e-02]	 2.1271324e-01
      12	-1.1348794e-02	 2.4167789e-01	 2.7339828e-03	 2.3831962e-01	[ 1.7002143e-02]	 1.9977450e-01


.. parsed-literal::

      13	-6.4479209e-03	 2.4072587e-01	 7.6250461e-03	 2.3754951e-01	[ 2.1793989e-02]	 2.0200682e-01


.. parsed-literal::

      14	 3.3329208e-02	 2.2569821e-01	 5.2174065e-02	 2.2296633e-01	[ 6.1854118e-02]	 4.3253112e-01
      15	 1.4387987e-01	 2.2068348e-01	 1.6605145e-01	 2.1753127e-01	[ 1.8015923e-01]	 2.0002651e-01


.. parsed-literal::

      16	 2.0924397e-01	 2.1441882e-01	 2.4004254e-01	 2.1168324e-01	[ 2.6052354e-01]	 2.1484542e-01


.. parsed-literal::

      17	 2.7855360e-01	 2.1338693e-01	 3.0841675e-01	 2.1085309e-01	[ 3.2849286e-01]	 2.3980904e-01
      18	 3.0201591e-01	 2.1144324e-01	 3.3184681e-01	 2.0971485e-01	[ 3.5182704e-01]	 1.9471335e-01


.. parsed-literal::

      19	 3.3059809e-01	 2.0868570e-01	 3.6018955e-01	 2.0818193e-01	[ 3.7782994e-01]	 2.0668602e-01
      20	 3.9083350e-01	 2.0425470e-01	 4.2252149e-01	 2.0392781e-01	[ 4.4409186e-01]	 1.8762779e-01


.. parsed-literal::

      21	 5.4606697e-01	 2.0626360e-01	 5.8278929e-01	 2.0222003e-01	[ 6.1484901e-01]	 2.0671177e-01


.. parsed-literal::

      22	 5.9577493e-01	 2.0771138e-01	 6.3481473e-01	 1.9958076e-01	[ 6.5294605e-01]	 2.0335412e-01


.. parsed-literal::

      23	 6.4554302e-01	 2.0044530e-01	 6.8216631e-01	 1.9454875e-01	[ 6.9744505e-01]	 2.1948528e-01


.. parsed-literal::

      24	 6.9336597e-01	 1.9646217e-01	 7.2957893e-01	 1.9195303e-01	[ 7.4143132e-01]	 2.1421981e-01


.. parsed-literal::

      25	 7.1388996e-01	 1.9515311e-01	 7.5037033e-01	 1.9165885e-01	[ 7.6014034e-01]	 3.2254076e-01
      26	 7.3771977e-01	 1.9584806e-01	 7.7480657e-01	 1.9361272e-01	[ 7.7773378e-01]	 1.8861127e-01


.. parsed-literal::

      27	 7.6141576e-01	 1.9651313e-01	 7.9875044e-01	 1.9576416e-01	[ 8.0294107e-01]	 2.0993781e-01


.. parsed-literal::

      28	 7.7746122e-01	 1.9637051e-01	 8.1473377e-01	 1.9667781e-01	[ 8.1765925e-01]	 2.0642328e-01


.. parsed-literal::

      29	 7.8627190e-01	 1.9809110e-01	 8.2403539e-01	 1.9952760e-01	[ 8.2498899e-01]	 2.0707941e-01


.. parsed-literal::

      30	 8.0392109e-01	 2.0031326e-01	 8.4234680e-01	 2.0425929e-01	[ 8.3881887e-01]	 2.0169592e-01
      31	 8.2289616e-01	 1.9890417e-01	 8.6139027e-01	 2.0272384e-01	[ 8.5723540e-01]	 1.9666243e-01


.. parsed-literal::

      32	 8.4208696e-01	 1.9604527e-01	 8.8083545e-01	 1.9869940e-01	[ 8.7656282e-01]	 2.1261024e-01


.. parsed-literal::

      33	 8.7201422e-01	 1.9257942e-01	 9.1135438e-01	 1.9397538e-01	[ 8.9923784e-01]	 2.1139884e-01


.. parsed-literal::

      34	 8.8926666e-01	 1.8757845e-01	 9.3053482e-01	 1.8685611e-01	[ 9.1695192e-01]	 2.0745778e-01


.. parsed-literal::

      35	 9.0867436e-01	 1.8694838e-01	 9.4940056e-01	 1.8544342e-01	[ 9.3821322e-01]	 2.0568347e-01


.. parsed-literal::

      36	 9.1854450e-01	 1.8688159e-01	 9.5909328e-01	 1.8616828e-01	[ 9.4392381e-01]	 2.1155405e-01


.. parsed-literal::

      37	 9.3503693e-01	 1.8538767e-01	 9.7590990e-01	 1.8568115e-01	[ 9.4914586e-01]	 2.0796990e-01


.. parsed-literal::

      38	 9.5100624e-01	 1.8409086e-01	 9.9238583e-01	 1.8412584e-01	[ 9.6428776e-01]	 2.1690989e-01
      39	 9.6301011e-01	 1.8345591e-01	 1.0059837e+00	 1.8248649e-01	[ 9.6889295e-01]	 1.8574452e-01


.. parsed-literal::

      40	 9.7601875e-01	 1.8237709e-01	 1.0187915e+00	 1.8150235e-01	[ 9.8457695e-01]	 2.0857406e-01


.. parsed-literal::

      41	 9.8400645e-01	 1.8155967e-01	 1.0268324e+00	 1.8081935e-01	[ 9.9148366e-01]	 2.0744205e-01


.. parsed-literal::

      42	 9.9474418e-01	 1.8065449e-01	 1.0379991e+00	 1.8044716e-01	[ 1.0011383e+00]	 2.0529485e-01
      43	 1.0010691e+00	 1.8055061e-01	 1.0451312e+00	 1.8045175e-01	[ 1.0024344e+00]	 1.8342829e-01


.. parsed-literal::

      44	 1.0090891e+00	 1.7991200e-01	 1.0531255e+00	 1.7969042e-01	[ 1.0143178e+00]	 1.8530035e-01


.. parsed-literal::

      45	 1.0158432e+00	 1.7973731e-01	 1.0602275e+00	 1.7929694e-01	[ 1.0236594e+00]	 2.1227789e-01


.. parsed-literal::

      46	 1.0228763e+00	 1.7953796e-01	 1.0673545e+00	 1.7909650e-01	[ 1.0320379e+00]	 2.1118212e-01


.. parsed-literal::

      47	 1.0371062e+00	 1.8028354e-01	 1.0819517e+00	 1.7976176e-01	[ 1.0472871e+00]	 2.1308255e-01


.. parsed-literal::

      48	 1.0455598e+00	 1.7981558e-01	 1.0905763e+00	 1.7927325e-01	[ 1.0518580e+00]	 3.1046033e-01


.. parsed-literal::

      49	 1.0530434e+00	 1.7928154e-01	 1.0977856e+00	 1.7874738e-01	[ 1.0619736e+00]	 2.1690321e-01


.. parsed-literal::

      50	 1.0603214e+00	 1.7846867e-01	 1.1050471e+00	 1.7829683e-01	[ 1.0676825e+00]	 2.0677280e-01


.. parsed-literal::

      51	 1.0673322e+00	 1.7673307e-01	 1.1122535e+00	 1.7630011e-01	[ 1.0776605e+00]	 2.1077847e-01
      52	 1.0751135e+00	 1.7492040e-01	 1.1202490e+00	 1.7485405e-01	[ 1.0836367e+00]	 1.9959283e-01


.. parsed-literal::

      53	 1.0831609e+00	 1.7385211e-01	 1.1286835e+00	 1.7364155e-01	[ 1.0942659e+00]	 2.1015811e-01
      54	 1.0932234e+00	 1.7195755e-01	 1.1390462e+00	 1.7151076e-01	[ 1.1123119e+00]	 1.9252443e-01


.. parsed-literal::

      55	 1.0988398e+00	 1.7061559e-01	 1.1447710e+00	 1.7026833e-01	[ 1.1185478e+00]	 2.1344876e-01


.. parsed-literal::

      56	 1.1051164e+00	 1.7010310e-01	 1.1509674e+00	 1.6970256e-01	[ 1.1242994e+00]	 2.1671820e-01


.. parsed-literal::

      57	 1.1138532e+00	 1.6888341e-01	 1.1598795e+00	 1.6831753e-01	[ 1.1319535e+00]	 2.1508908e-01


.. parsed-literal::

      58	 1.1208069e+00	 1.6804976e-01	 1.1671113e+00	 1.6723562e-01	[ 1.1383321e+00]	 2.0788026e-01
      59	 1.1291534e+00	 1.6552204e-01	 1.1761740e+00	 1.6402494e-01	[ 1.1526158e+00]	 1.7114115e-01


.. parsed-literal::

      60	 1.1371886e+00	 1.6429231e-01	 1.1842781e+00	 1.6258367e-01	[ 1.1581337e+00]	 2.1369243e-01


.. parsed-literal::

      61	 1.1413882e+00	 1.6344667e-01	 1.1884645e+00	 1.6175305e-01	[ 1.1609110e+00]	 2.2576809e-01
      62	 1.1477135e+00	 1.6126714e-01	 1.1950366e+00	 1.5942522e-01	[ 1.1667487e+00]	 1.9846010e-01


.. parsed-literal::

      63	 1.1522113e+00	 1.5968484e-01	 1.1998979e+00	 1.5812261e-01	[ 1.1690462e+00]	 2.1660972e-01


.. parsed-literal::

      64	 1.1586971e+00	 1.5867845e-01	 1.2063917e+00	 1.5689481e-01	[ 1.1730327e+00]	 2.1594119e-01


.. parsed-literal::

      65	 1.1630744e+00	 1.5835152e-01	 1.2108086e+00	 1.5655539e-01	[ 1.1745296e+00]	 2.0978928e-01


.. parsed-literal::

      66	 1.1684863e+00	 1.5814887e-01	 1.2163919e+00	 1.5644462e-01	  1.1735435e+00 	 2.2267985e-01
      67	 1.1770204e+00	 1.5819520e-01	 1.2253125e+00	 1.5636459e-01	[ 1.1764419e+00]	 1.8366981e-01


.. parsed-literal::

      68	 1.1813841e+00	 1.5873265e-01	 1.2301109e+00	 1.5679959e-01	  1.1662348e+00 	 2.0735121e-01


.. parsed-literal::

      69	 1.1859333e+00	 1.5809128e-01	 1.2343574e+00	 1.5623595e-01	[ 1.1768522e+00]	 2.1567059e-01


.. parsed-literal::

      70	 1.1912296e+00	 1.5704117e-01	 1.2396729e+00	 1.5514454e-01	[ 1.1845001e+00]	 2.0591044e-01


.. parsed-literal::

      71	 1.1969700e+00	 1.5632339e-01	 1.2456920e+00	 1.5456143e-01	[ 1.1860429e+00]	 2.1331453e-01


.. parsed-literal::

      72	 1.2022805e+00	 1.5594544e-01	 1.2511229e+00	 1.5457471e-01	  1.1803592e+00 	 2.0192504e-01


.. parsed-literal::

      73	 1.2064298e+00	 1.5550926e-01	 1.2552707e+00	 1.5420238e-01	  1.1771113e+00 	 2.1120501e-01


.. parsed-literal::

      74	 1.2121214e+00	 1.5469420e-01	 1.2610514e+00	 1.5339178e-01	  1.1721766e+00 	 2.1374750e-01


.. parsed-literal::

      75	 1.2171929e+00	 1.5390970e-01	 1.2662892e+00	 1.5271299e-01	  1.1678712e+00 	 2.2019601e-01
      76	 1.2217685e+00	 1.5287290e-01	 1.2709719e+00	 1.5154266e-01	  1.1678763e+00 	 1.9185352e-01


.. parsed-literal::

      77	 1.2252745e+00	 1.5206923e-01	 1.2744743e+00	 1.5074477e-01	  1.1743581e+00 	 2.1290517e-01


.. parsed-literal::

      78	 1.2285112e+00	 1.5121627e-01	 1.2777675e+00	 1.4969632e-01	  1.1799821e+00 	 2.0885921e-01
      79	 1.2346872e+00	 1.4970916e-01	 1.2841951e+00	 1.4738161e-01	[ 1.1896975e+00]	 1.9902301e-01


.. parsed-literal::

      80	 1.2396966e+00	 1.4879819e-01	 1.2892812e+00	 1.4600648e-01	  1.1896878e+00 	 1.9393516e-01


.. parsed-literal::

      81	 1.2430955e+00	 1.4861703e-01	 1.2926889e+00	 1.4566779e-01	  1.1886724e+00 	 2.1713448e-01
      82	 1.2479000e+00	 1.4842564e-01	 1.2976729e+00	 1.4527398e-01	  1.1838099e+00 	 1.9779634e-01


.. parsed-literal::

      83	 1.2514413e+00	 1.4829070e-01	 1.3013606e+00	 1.4563873e-01	  1.1789872e+00 	 2.1304965e-01


.. parsed-literal::

      84	 1.2548971e+00	 1.4812107e-01	 1.3047910e+00	 1.4542927e-01	  1.1821616e+00 	 2.0743561e-01


.. parsed-literal::

      85	 1.2592017e+00	 1.4766282e-01	 1.3092473e+00	 1.4507664e-01	  1.1829537e+00 	 2.1348977e-01


.. parsed-literal::

      86	 1.2619977e+00	 1.4765461e-01	 1.3121335e+00	 1.4510442e-01	  1.1825751e+00 	 2.0885515e-01
      87	 1.2666984e+00	 1.4763006e-01	 1.3170156e+00	 1.4519616e-01	  1.1808479e+00 	 1.8541169e-01


.. parsed-literal::

      88	 1.2712525e+00	 1.4755436e-01	 1.3218006e+00	 1.4534552e-01	  1.1729466e+00 	 2.1471810e-01
      89	 1.2752676e+00	 1.4757353e-01	 1.3258945e+00	 1.4525942e-01	  1.1724589e+00 	 1.9826198e-01


.. parsed-literal::

      90	 1.2779366e+00	 1.4749999e-01	 1.3285139e+00	 1.4515114e-01	  1.1773774e+00 	 2.1409106e-01


.. parsed-literal::

      91	 1.2808032e+00	 1.4725523e-01	 1.3314249e+00	 1.4481700e-01	  1.1770078e+00 	 2.0948458e-01


.. parsed-literal::

      92	 1.2845509e+00	 1.4668881e-01	 1.3355123e+00	 1.4388286e-01	  1.1788986e+00 	 2.0749378e-01


.. parsed-literal::

      93	 1.2881708e+00	 1.4625112e-01	 1.3392332e+00	 1.4336171e-01	  1.1781311e+00 	 2.1776843e-01
      94	 1.2910900e+00	 1.4572094e-01	 1.3422036e+00	 1.4272097e-01	  1.1786570e+00 	 2.0017242e-01


.. parsed-literal::

      95	 1.2948164e+00	 1.4499449e-01	 1.3460435e+00	 1.4166514e-01	  1.1831219e+00 	 1.8827105e-01
      96	 1.2982281e+00	 1.4439768e-01	 1.3496784e+00	 1.4071245e-01	  1.1800572e+00 	 1.9524336e-01


.. parsed-literal::

      97	 1.3017900e+00	 1.4398412e-01	 1.3531649e+00	 1.4014413e-01	  1.1856166e+00 	 2.0597458e-01


.. parsed-literal::

      98	 1.3054104e+00	 1.4357256e-01	 1.3567882e+00	 1.3964558e-01	  1.1875532e+00 	 2.0590186e-01


.. parsed-literal::

      99	 1.3087700e+00	 1.4326015e-01	 1.3601556e+00	 1.3939624e-01	  1.1886731e+00 	 2.0819139e-01


.. parsed-literal::

     100	 1.3120193e+00	 1.4285375e-01	 1.3635415e+00	 1.3956446e-01	[ 1.1929768e+00]	 2.0329833e-01


.. parsed-literal::

     101	 1.3161523e+00	 1.4266062e-01	 1.3676773e+00	 1.3938113e-01	  1.1926888e+00 	 2.0213652e-01
     102	 1.3182106e+00	 1.4263220e-01	 1.3697179e+00	 1.3945838e-01	[ 1.1930945e+00]	 1.9396806e-01


.. parsed-literal::

     103	 1.3212548e+00	 1.4236275e-01	 1.3728785e+00	 1.3935786e-01	  1.1924195e+00 	 2.0659542e-01


.. parsed-literal::

     104	 1.3227853e+00	 1.4255506e-01	 1.3746216e+00	 1.3979835e-01	  1.1858683e+00 	 2.0978069e-01


.. parsed-literal::

     105	 1.3266499e+00	 1.4217065e-01	 1.3784244e+00	 1.3937767e-01	  1.1896333e+00 	 2.0495033e-01


.. parsed-literal::

     106	 1.3286253e+00	 1.4194734e-01	 1.3804041e+00	 1.3909941e-01	  1.1910496e+00 	 2.2081566e-01


.. parsed-literal::

     107	 1.3308153e+00	 1.4174323e-01	 1.3826497e+00	 1.3888088e-01	  1.1907436e+00 	 2.1397853e-01


.. parsed-literal::

     108	 1.3348725e+00	 1.4148955e-01	 1.3868403e+00	 1.3862174e-01	  1.1877984e+00 	 2.0419502e-01


.. parsed-literal::

     109	 1.3380332e+00	 1.4115104e-01	 1.3901909e+00	 1.3833875e-01	  1.1818389e+00 	 3.3155155e-01


.. parsed-literal::

     110	 1.3421097e+00	 1.4097929e-01	 1.3944354e+00	 1.3816537e-01	  1.1727032e+00 	 2.0634961e-01


.. parsed-literal::

     111	 1.3448194e+00	 1.4084346e-01	 1.3972280e+00	 1.3800811e-01	  1.1637354e+00 	 2.1344709e-01


.. parsed-literal::

     112	 1.3474392e+00	 1.4054534e-01	 1.3999473e+00	 1.3773846e-01	  1.1573065e+00 	 2.0993781e-01


.. parsed-literal::

     113	 1.3502919e+00	 1.4013160e-01	 1.4028124e+00	 1.3730845e-01	  1.1577052e+00 	 2.1502519e-01


.. parsed-literal::

     114	 1.3538418e+00	 1.3948515e-01	 1.4065800e+00	 1.3642939e-01	  1.1565909e+00 	 2.1044517e-01


.. parsed-literal::

     115	 1.3567799e+00	 1.3924500e-01	 1.4095695e+00	 1.3616789e-01	  1.1653001e+00 	 2.0175672e-01


.. parsed-literal::

     116	 1.3588110e+00	 1.3919448e-01	 1.4115619e+00	 1.3614328e-01	  1.1684083e+00 	 2.1068788e-01


.. parsed-literal::

     117	 1.3620965e+00	 1.3909181e-01	 1.4149086e+00	 1.3609022e-01	  1.1708423e+00 	 2.1264935e-01


.. parsed-literal::

     118	 1.3647759e+00	 1.3922968e-01	 1.4176846e+00	 1.3630606e-01	  1.1645914e+00 	 2.1412802e-01


.. parsed-literal::

     119	 1.3679043e+00	 1.3886397e-01	 1.4208376e+00	 1.3603257e-01	  1.1612769e+00 	 2.0918584e-01


.. parsed-literal::

     120	 1.3713503e+00	 1.3830810e-01	 1.4242741e+00	 1.3567426e-01	  1.1565044e+00 	 2.1438909e-01


.. parsed-literal::

     121	 1.3737858e+00	 1.3783257e-01	 1.4266797e+00	 1.3533857e-01	  1.1549666e+00 	 2.0615172e-01


.. parsed-literal::

     122	 1.3764915e+00	 1.3708442e-01	 1.4294155e+00	 1.3496350e-01	  1.1527576e+00 	 2.1278358e-01


.. parsed-literal::

     123	 1.3799877e+00	 1.3674585e-01	 1.4328547e+00	 1.3468813e-01	  1.1506763e+00 	 2.1637487e-01


.. parsed-literal::

     124	 1.3816029e+00	 1.3674269e-01	 1.4344696e+00	 1.3468111e-01	  1.1503263e+00 	 2.2244120e-01


.. parsed-literal::

     125	 1.3843776e+00	 1.3668307e-01	 1.4373591e+00	 1.3474605e-01	  1.1434956e+00 	 2.1048713e-01
     126	 1.3862570e+00	 1.3635360e-01	 1.4394455e+00	 1.3463269e-01	  1.1275083e+00 	 1.8954587e-01


.. parsed-literal::

     127	 1.3887859e+00	 1.3629028e-01	 1.4419473e+00	 1.3468047e-01	  1.1282543e+00 	 2.1193933e-01
     128	 1.3909368e+00	 1.3608306e-01	 1.4441349e+00	 1.3470184e-01	  1.1237292e+00 	 1.9549656e-01


.. parsed-literal::

     129	 1.3927783e+00	 1.3587798e-01	 1.4460004e+00	 1.3470384e-01	  1.1232155e+00 	 2.1172643e-01


.. parsed-literal::

     130	 1.3962164e+00	 1.3538402e-01	 1.4495609e+00	 1.3476161e-01	  1.1178965e+00 	 2.1421337e-01
     131	 1.3972564e+00	 1.3482698e-01	 1.4509304e+00	 1.3485355e-01	  1.1165089e+00 	 1.9835877e-01


.. parsed-literal::

     132	 1.4006826e+00	 1.3485941e-01	 1.4541557e+00	 1.3476100e-01	  1.1211908e+00 	 1.9936562e-01


.. parsed-literal::

     133	 1.4023380e+00	 1.3478492e-01	 1.4558101e+00	 1.3475256e-01	  1.1201804e+00 	 2.1199489e-01
     134	 1.4049096e+00	 1.3455371e-01	 1.4584787e+00	 1.3475467e-01	  1.1159236e+00 	 1.8028307e-01


.. parsed-literal::

     135	 1.4082500e+00	 1.3409682e-01	 1.4619875e+00	 1.3472181e-01	  1.1039284e+00 	 2.1140027e-01


.. parsed-literal::

     136	 1.4104708e+00	 1.3374129e-01	 1.4643513e+00	 1.3476312e-01	  1.0918453e+00 	 3.2155848e-01


.. parsed-literal::

     137	 1.4128550e+00	 1.3345052e-01	 1.4667555e+00	 1.3476725e-01	  1.0864973e+00 	 2.0219874e-01


.. parsed-literal::

     138	 1.4146269e+00	 1.3326152e-01	 1.4684913e+00	 1.3477851e-01	  1.0854038e+00 	 2.0372510e-01
     139	 1.4168548e+00	 1.3308025e-01	 1.4707173e+00	 1.3479191e-01	  1.0819644e+00 	 1.8630719e-01


.. parsed-literal::

     140	 1.4186139e+00	 1.3300740e-01	 1.4725148e+00	 1.3497483e-01	  1.0814615e+00 	 3.0103993e-01


.. parsed-literal::

     141	 1.4212114e+00	 1.3272503e-01	 1.4752519e+00	 1.3499376e-01	  1.0700455e+00 	 2.1485257e-01


.. parsed-literal::

     142	 1.4228430e+00	 1.3266448e-01	 1.4768914e+00	 1.3493550e-01	  1.0686736e+00 	 2.0788789e-01


.. parsed-literal::

     143	 1.4248138e+00	 1.3252069e-01	 1.4789328e+00	 1.3484001e-01	  1.0624154e+00 	 2.0211673e-01
     144	 1.4265850e+00	 1.3218855e-01	 1.4809250e+00	 1.3500817e-01	  1.0353321e+00 	 2.0424533e-01


.. parsed-literal::

     145	 1.4290662e+00	 1.3204963e-01	 1.4833563e+00	 1.3495276e-01	  1.0352816e+00 	 1.9820642e-01
     146	 1.4309381e+00	 1.3183900e-01	 1.4852300e+00	 1.3503852e-01	  1.0289036e+00 	 1.9775438e-01


.. parsed-literal::

     147	 1.4324406e+00	 1.3168546e-01	 1.4867323e+00	 1.3516112e-01	  1.0239686e+00 	 2.0410061e-01
     148	 1.4356445e+00	 1.3134178e-01	 1.4899706e+00	 1.3547071e-01	  1.0106897e+00 	 1.9451809e-01


.. parsed-literal::

     149	 1.4377683e+00	 1.3121253e-01	 1.4921143e+00	 1.3566823e-01	  1.0074526e+00 	 3.0929136e-01


.. parsed-literal::

     150	 1.4401649e+00	 1.3105522e-01	 1.4945302e+00	 1.3583446e-01	  1.0011746e+00 	 2.0673943e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1 s, total: 2min 8s
    Wall time: 32.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f81ba5c3f70>



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
    CPU times: user 1.76 s, sys: 42 ms, total: 1.8 s
    Wall time: 598 ms


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

