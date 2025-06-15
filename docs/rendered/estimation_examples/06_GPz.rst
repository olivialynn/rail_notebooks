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
       1	-3.5238021e-01	 3.2339000e-01	-3.4275900e-01	 3.0890970e-01	[-3.1621646e-01]	 4.6355295e-01


.. parsed-literal::

       2	-2.8188187e-01	 3.1277538e-01	-2.5745475e-01	 2.9878620e-01	[-2.1570145e-01]	 2.3337603e-01


.. parsed-literal::

       3	-2.3487881e-01	 2.8934372e-01	-1.8884061e-01	 2.8282107e-01	[-1.6062217e-01]	 2.7885127e-01
       4	-1.9770006e-01	 2.6577069e-01	-1.5436347e-01	 2.6792820e-01	[-1.5883011e-01]	 1.9780850e-01


.. parsed-literal::

       5	-1.0150510e-01	 2.5734234e-01	-6.7633060e-02	 2.5420518e-01	[-5.2274036e-02]	 1.7565322e-01


.. parsed-literal::

       6	-7.3468494e-02	 2.5281129e-01	-4.4342737e-02	 2.4969767e-01	[-3.1567077e-02]	 2.0755911e-01


.. parsed-literal::

       7	-5.4633286e-02	 2.4969203e-01	-3.1656316e-02	 2.4615003e-01	[-1.6981841e-02]	 2.0794940e-01
       8	-4.3295620e-02	 2.4778607e-01	-2.3702299e-02	 2.4411918e-01	[-8.6897506e-03]	 1.8956780e-01


.. parsed-literal::

       9	-3.0669662e-02	 2.4540238e-01	-1.3515630e-02	 2.4189410e-01	[ 3.5871937e-04]	 2.0416260e-01


.. parsed-literal::

      10	-1.8718980e-02	 2.4288154e-01	-3.0542784e-03	 2.3939432e-01	[ 1.1175188e-02]	 2.0587778e-01


.. parsed-literal::

      11	-1.5300805e-02	 2.4235375e-01	-1.0205904e-03	 2.3921527e-01	  1.0236287e-02 	 2.0104408e-01


.. parsed-literal::

      12	-1.0113010e-02	 2.4156226e-01	 4.0312480e-03	 2.3856490e-01	[ 1.5799583e-02]	 2.2080922e-01


.. parsed-literal::

      13	-7.1459595e-03	 2.4092360e-01	 7.0389928e-03	 2.3816888e-01	[ 1.8383358e-02]	 2.2037888e-01
      14	-2.7867090e-03	 2.3990918e-01	 1.1985172e-02	 2.3767935e-01	[ 2.1980461e-02]	 1.9966817e-01


.. parsed-literal::

      15	 1.2376157e-01	 2.2703313e-01	 1.4544108e-01	 2.2270402e-01	[ 1.5419047e-01]	 3.2650638e-01


.. parsed-literal::

      16	 1.5958749e-01	 2.2284827e-01	 1.8255756e-01	 2.2125873e-01	[ 1.8842260e-01]	 3.3468843e-01


.. parsed-literal::

      17	 2.1675691e-01	 2.1838626e-01	 2.4128376e-01	 2.1475577e-01	[ 2.5027694e-01]	 2.0447421e-01
      18	 3.0637680e-01	 2.1479208e-01	 3.3677319e-01	 2.1306213e-01	[ 3.5013410e-01]	 1.8654537e-01


.. parsed-literal::

      19	 3.5009046e-01	 2.0989650e-01	 3.8148531e-01	 2.0941137e-01	[ 3.8953328e-01]	 2.1609640e-01


.. parsed-literal::

      20	 4.0887182e-01	 2.0472180e-01	 4.4102700e-01	 2.0343899e-01	[ 4.4653113e-01]	 2.1384072e-01


.. parsed-literal::

      21	 4.7374338e-01	 2.0529067e-01	 5.0652012e-01	 2.0379330e-01	[ 5.1028307e-01]	 2.0991778e-01
      22	 5.4706639e-01	 2.0544107e-01	 5.8153649e-01	 2.0324832e-01	[ 5.7831413e-01]	 2.0024848e-01


.. parsed-literal::

      23	 6.0154522e-01	 2.0164440e-01	 6.3797119e-01	 1.9995162e-01	[ 6.2618188e-01]	 2.0861101e-01
      24	 6.1459894e-01	 2.0740250e-01	 6.5284424e-01	 2.0094939e-01	[ 6.5069139e-01]	 1.7559171e-01


.. parsed-literal::

      25	 6.4586805e-01	 2.0169437e-01	 6.8257988e-01	 1.9664630e-01	[ 6.7817451e-01]	 2.1197033e-01


.. parsed-literal::

      26	 6.6308703e-01	 2.0076613e-01	 6.9944941e-01	 1.9762296e-01	[ 6.9199858e-01]	 2.0839572e-01
      27	 6.8183631e-01	 2.0057366e-01	 7.1793164e-01	 2.0144549e-01	[ 7.0540556e-01]	 2.0609379e-01


.. parsed-literal::

      28	 6.9426484e-01	 2.0454880e-01	 7.3020593e-01	 2.0293761e-01	[ 7.1966719e-01]	 2.0975327e-01


.. parsed-literal::

      29	 7.2238638e-01	 2.0214474e-01	 7.5908295e-01	 2.0325416e-01	[ 7.4978033e-01]	 2.1060061e-01


.. parsed-literal::

      30	 7.3857541e-01	 2.0050086e-01	 7.7557579e-01	 2.0147479e-01	[ 7.6649725e-01]	 2.0170474e-01
      31	 7.5223845e-01	 2.0035076e-01	 7.8944663e-01	 1.9837936e-01	[ 7.8544346e-01]	 2.0288992e-01


.. parsed-literal::

      32	 7.7304444e-01	 2.0111183e-01	 8.1076140e-01	 1.9637072e-01	[ 8.1627530e-01]	 2.1908307e-01
      33	 7.9372443e-01	 2.0053241e-01	 8.3212735e-01	 1.9625481e-01	[ 8.3630044e-01]	 1.9061184e-01


.. parsed-literal::

      34	 8.1239506e-01	 2.0005331e-01	 8.5102896e-01	 1.9621025e-01	[ 8.5484071e-01]	 1.8738508e-01
      35	 8.3280839e-01	 1.9629744e-01	 8.7175314e-01	 1.9229298e-01	[ 8.7398019e-01]	 1.9635892e-01


.. parsed-literal::

      36	 8.5706147e-01	 1.9342245e-01	 8.9595903e-01	 1.8925207e-01	[ 8.9410008e-01]	 2.0387888e-01
      37	 8.7273444e-01	 1.9113690e-01	 9.1192739e-01	 1.8815657e-01	[ 9.1216347e-01]	 2.0221329e-01


.. parsed-literal::

      38	 8.8521459e-01	 1.9178614e-01	 9.2447132e-01	 1.8876403e-01	[ 9.2123514e-01]	 2.0379543e-01


.. parsed-literal::

      39	 9.0322324e-01	 1.9335252e-01	 9.4325992e-01	 1.9003817e-01	[ 9.3320783e-01]	 2.1789408e-01


.. parsed-literal::

      40	 9.2033728e-01	 1.9424650e-01	 9.6174253e-01	 1.9061981e-01	[ 9.3812588e-01]	 2.1305561e-01
      41	 9.3474323e-01	 1.9454997e-01	 9.7710502e-01	 1.9029882e-01	[ 9.4240715e-01]	 1.7800570e-01


.. parsed-literal::

      42	 9.4476619e-01	 1.9336871e-01	 9.8714583e-01	 1.8947543e-01	[ 9.4738471e-01]	 2.2382331e-01


.. parsed-literal::

      43	 9.5849639e-01	 1.9129342e-01	 1.0010662e+00	 1.8766537e-01	[ 9.5201010e-01]	 2.1197534e-01


.. parsed-literal::

      44	 9.7091306e-01	 1.8961958e-01	 1.0137694e+00	 1.8552942e-01	[ 9.6356921e-01]	 2.1047521e-01


.. parsed-literal::

      45	 9.8130456e-01	 1.8961046e-01	 1.0249625e+00	 1.8494837e-01	  9.6057082e-01 	 2.1876645e-01


.. parsed-literal::

      46	 9.9181835e-01	 1.8853820e-01	 1.0360351e+00	 1.8328898e-01	  9.5610947e-01 	 2.0879340e-01
      47	 1.0009039e+00	 1.8750155e-01	 1.0455738e+00	 1.8266094e-01	  9.4895174e-01 	 1.9831681e-01


.. parsed-literal::

      48	 1.0096646e+00	 1.8643672e-01	 1.0546580e+00	 1.8171933e-01	  9.4507269e-01 	 2.1370983e-01
      49	 1.0192309e+00	 1.8593888e-01	 1.0648171e+00	 1.8121599e-01	  9.2044035e-01 	 2.0301914e-01


.. parsed-literal::

      50	 1.0272828e+00	 1.8490235e-01	 1.0727717e+00	 1.8054028e-01	  9.3214082e-01 	 1.8727398e-01


.. parsed-literal::

      51	 1.0330152e+00	 1.8390680e-01	 1.0782382e+00	 1.7984171e-01	  9.3668496e-01 	 2.2108269e-01


.. parsed-literal::

      52	 1.0399831e+00	 1.8212040e-01	 1.0852309e+00	 1.7860877e-01	  9.3976408e-01 	 2.1160769e-01


.. parsed-literal::

      53	 1.0448357e+00	 1.8171771e-01	 1.0900627e+00	 1.7774753e-01	  9.3770834e-01 	 2.0625043e-01


.. parsed-literal::

      54	 1.0494035e+00	 1.8149320e-01	 1.0947278e+00	 1.7722700e-01	  9.4342426e-01 	 2.1323872e-01


.. parsed-literal::

      55	 1.0575031e+00	 1.8042915e-01	 1.1034843e+00	 1.7592222e-01	  9.4660593e-01 	 2.1815753e-01


.. parsed-literal::

      56	 1.0640813e+00	 1.7968424e-01	 1.1104703e+00	 1.7416524e-01	  9.5109733e-01 	 2.2613811e-01


.. parsed-literal::

      57	 1.0712086e+00	 1.7855045e-01	 1.1176975e+00	 1.7299520e-01	  9.5946418e-01 	 2.2221661e-01
      58	 1.0778382e+00	 1.7740685e-01	 1.1241937e+00	 1.7185433e-01	[ 9.7813169e-01]	 1.9645786e-01


.. parsed-literal::

      59	 1.0817288e+00	 1.7676279e-01	 1.1280397e+00	 1.7064963e-01	[ 9.9221249e-01]	 2.0759678e-01


.. parsed-literal::

      60	 1.0859781e+00	 1.7637811e-01	 1.1321182e+00	 1.7020000e-01	[ 1.0049892e+00]	 2.1066737e-01


.. parsed-literal::

      61	 1.0920103e+00	 1.7519981e-01	 1.1382256e+00	 1.6866365e-01	[ 1.0215934e+00]	 2.2063208e-01


.. parsed-literal::

      62	 1.0978232e+00	 1.7437288e-01	 1.1441865e+00	 1.6738629e-01	[ 1.0339917e+00]	 2.1186662e-01
      63	 1.1032456e+00	 1.7281582e-01	 1.1499037e+00	 1.6554975e-01	  1.0339555e+00 	 1.8451667e-01


.. parsed-literal::

      64	 1.1080144e+00	 1.7185993e-01	 1.1546980e+00	 1.6457491e-01	[ 1.0343786e+00]	 2.1350670e-01


.. parsed-literal::

      65	 1.1117131e+00	 1.7118883e-01	 1.1584396e+00	 1.6418430e-01	  1.0273678e+00 	 2.1831918e-01


.. parsed-literal::

      66	 1.1158308e+00	 1.6990463e-01	 1.1626661e+00	 1.6331094e-01	  1.0263009e+00 	 2.3073387e-01


.. parsed-literal::

      67	 1.1204667e+00	 1.6849600e-01	 1.1675675e+00	 1.6252538e-01	  1.0204426e+00 	 2.2133970e-01
      68	 1.1263989e+00	 1.6767704e-01	 1.1734402e+00	 1.6216441e-01	  1.0284651e+00 	 1.7936754e-01


.. parsed-literal::

      69	 1.1298172e+00	 1.6745945e-01	 1.1768603e+00	 1.6212042e-01	  1.0330171e+00 	 2.1078920e-01


.. parsed-literal::

      70	 1.1352265e+00	 1.6689052e-01	 1.1824167e+00	 1.6211174e-01	  1.0342803e+00 	 2.1826458e-01


.. parsed-literal::

      71	 1.1436456e+00	 1.6539910e-01	 1.1912258e+00	 1.6159084e-01	  1.0257578e+00 	 2.1534562e-01


.. parsed-literal::

      72	 1.1487860e+00	 1.6437725e-01	 1.1966545e+00	 1.6157695e-01	[ 1.0345668e+00]	 3.2224894e-01


.. parsed-literal::

      73	 1.1533376e+00	 1.6348025e-01	 1.2013163e+00	 1.6114530e-01	  1.0294510e+00 	 2.2342801e-01


.. parsed-literal::

      74	 1.1586431e+00	 1.6214573e-01	 1.2067684e+00	 1.5998237e-01	[ 1.0349741e+00]	 2.1670508e-01


.. parsed-literal::

      75	 1.1644826e+00	 1.6045639e-01	 1.2128217e+00	 1.5830810e-01	[ 1.0417164e+00]	 2.1996117e-01
      76	 1.1705058e+00	 1.5928181e-01	 1.2190579e+00	 1.5721962e-01	[ 1.0497911e+00]	 1.9113755e-01


.. parsed-literal::

      77	 1.1749057e+00	 1.5860477e-01	 1.2234826e+00	 1.5646403e-01	[ 1.0598170e+00]	 2.1412563e-01
      78	 1.1793823e+00	 1.5799297e-01	 1.2280229e+00	 1.5566442e-01	[ 1.0684265e+00]	 1.8945169e-01


.. parsed-literal::

      79	 1.1850084e+00	 1.5665723e-01	 1.2338152e+00	 1.5409872e-01	[ 1.0744405e+00]	 2.0457053e-01


.. parsed-literal::

      80	 1.1898710e+00	 1.5607971e-01	 1.2387148e+00	 1.5315326e-01	[ 1.0866471e+00]	 2.2603226e-01


.. parsed-literal::

      81	 1.1932967e+00	 1.5590348e-01	 1.2420194e+00	 1.5268844e-01	[ 1.0909252e+00]	 2.1113729e-01
      82	 1.1981051e+00	 1.5497664e-01	 1.2468301e+00	 1.5202543e-01	[ 1.0911944e+00]	 1.8937492e-01


.. parsed-literal::

      83	 1.2010667e+00	 1.5447183e-01	 1.2499835e+00	 1.5167278e-01	[ 1.1001711e+00]	 2.0759749e-01
      84	 1.2052299e+00	 1.5397627e-01	 1.2541082e+00	 1.5156393e-01	[ 1.1014925e+00]	 1.7393398e-01


.. parsed-literal::

      85	 1.2094720e+00	 1.5342628e-01	 1.2584913e+00	 1.5157840e-01	[ 1.1054150e+00]	 2.1493125e-01


.. parsed-literal::

      86	 1.2126029e+00	 1.5322783e-01	 1.2616554e+00	 1.5167058e-01	[ 1.1054878e+00]	 2.1132445e-01


.. parsed-literal::

      87	 1.2165716e+00	 1.5304823e-01	 1.2659867e+00	 1.5175831e-01	[ 1.1108829e+00]	 2.2271681e-01


.. parsed-literal::

      88	 1.2224662e+00	 1.5280946e-01	 1.2716424e+00	 1.5165046e-01	  1.1002519e+00 	 2.0402193e-01
      89	 1.2245920e+00	 1.5261096e-01	 1.2736742e+00	 1.5118797e-01	  1.1019679e+00 	 1.9197249e-01


.. parsed-literal::

      90	 1.2295126e+00	 1.5216379e-01	 1.2786703e+00	 1.4997623e-01	  1.1049090e+00 	 2.1047449e-01
      91	 1.2315827e+00	 1.5301324e-01	 1.2810677e+00	 1.5010914e-01	  1.0936106e+00 	 1.8031287e-01


.. parsed-literal::

      92	 1.2360939e+00	 1.5236918e-01	 1.2855851e+00	 1.4949065e-01	  1.0964526e+00 	 2.0205474e-01


.. parsed-literal::

      93	 1.2389696e+00	 1.5214655e-01	 1.2885461e+00	 1.4934998e-01	  1.0955694e+00 	 2.0791745e-01
      94	 1.2420325e+00	 1.5221434e-01	 1.2917733e+00	 1.4929061e-01	  1.0920888e+00 	 1.7426038e-01


.. parsed-literal::

      95	 1.2468666e+00	 1.5215943e-01	 1.2968470e+00	 1.4905932e-01	  1.0922826e+00 	 2.2967601e-01


.. parsed-literal::

      96	 1.2511367e+00	 1.5247530e-01	 1.3014250e+00	 1.4875852e-01	  1.0849922e+00 	 2.1292138e-01


.. parsed-literal::

      97	 1.2542143e+00	 1.5219124e-01	 1.3044327e+00	 1.4843300e-01	  1.0874809e+00 	 2.0451784e-01


.. parsed-literal::

      98	 1.2581761e+00	 1.5076412e-01	 1.3085001e+00	 1.4710572e-01	  1.0880535e+00 	 2.0774460e-01
      99	 1.2610459e+00	 1.5059294e-01	 1.3115076e+00	 1.4675035e-01	  1.0999869e+00 	 1.7223191e-01


.. parsed-literal::

     100	 1.2634035e+00	 1.5036471e-01	 1.3138557e+00	 1.4675584e-01	  1.1000821e+00 	 2.1773124e-01
     101	 1.2675961e+00	 1.4941929e-01	 1.3182901e+00	 1.4631376e-01	  1.1045717e+00 	 1.8197656e-01


.. parsed-literal::

     102	 1.2705185e+00	 1.4887737e-01	 1.3213203e+00	 1.4601262e-01	[ 1.1113549e+00]	 2.1605253e-01
     103	 1.2749162e+00	 1.4761673e-01	 1.3258578e+00	 1.4525689e-01	[ 1.1266312e+00]	 1.7921758e-01


.. parsed-literal::

     104	 1.2783468e+00	 1.4728040e-01	 1.3292458e+00	 1.4484307e-01	[ 1.1349356e+00]	 2.0584655e-01


.. parsed-literal::

     105	 1.2805823e+00	 1.4715038e-01	 1.3313037e+00	 1.4466478e-01	[ 1.1369172e+00]	 2.1047592e-01


.. parsed-literal::

     106	 1.2853435e+00	 1.4614967e-01	 1.3359899e+00	 1.4338834e-01	[ 1.1432210e+00]	 2.2249699e-01
     107	 1.2863230e+00	 1.4570759e-01	 1.3371734e+00	 1.4256643e-01	[ 1.1505492e+00]	 1.8878365e-01


.. parsed-literal::

     108	 1.2905276e+00	 1.4538983e-01	 1.3412893e+00	 1.4237717e-01	  1.1499227e+00 	 2.0991397e-01


.. parsed-literal::

     109	 1.2927339e+00	 1.4514503e-01	 1.3435768e+00	 1.4211550e-01	[ 1.1512105e+00]	 2.1276426e-01


.. parsed-literal::

     110	 1.2956820e+00	 1.4465694e-01	 1.3466949e+00	 1.4145560e-01	[ 1.1557759e+00]	 2.1939826e-01
     111	 1.2982123e+00	 1.4463480e-01	 1.3494078e+00	 1.4102462e-01	[ 1.1630071e+00]	 1.7471313e-01


.. parsed-literal::

     112	 1.3011707e+00	 1.4420608e-01	 1.3523752e+00	 1.4032464e-01	[ 1.1709959e+00]	 2.0680737e-01


.. parsed-literal::

     113	 1.3045580e+00	 1.4401809e-01	 1.3556666e+00	 1.3988930e-01	[ 1.1769643e+00]	 2.1088004e-01


.. parsed-literal::

     114	 1.3079413e+00	 1.4377339e-01	 1.3590497e+00	 1.3936778e-01	[ 1.1791602e+00]	 2.0731592e-01


.. parsed-literal::

     115	 1.3102184e+00	 1.4354851e-01	 1.3614580e+00	 1.3908547e-01	[ 1.1819536e+00]	 2.9941154e-01
     116	 1.3133526e+00	 1.4347032e-01	 1.3646732e+00	 1.3903065e-01	  1.1769719e+00 	 1.9852304e-01


.. parsed-literal::

     117	 1.3156864e+00	 1.4326537e-01	 1.3671254e+00	 1.3899360e-01	  1.1726550e+00 	 2.1252036e-01


.. parsed-literal::

     118	 1.3189183e+00	 1.4312264e-01	 1.3705709e+00	 1.3919321e-01	  1.1681434e+00 	 2.1695375e-01
     119	 1.3219024e+00	 1.4261797e-01	 1.3737400e+00	 1.3924167e-01	  1.1632880e+00 	 2.0750403e-01


.. parsed-literal::

     120	 1.3252460e+00	 1.4235218e-01	 1.3771368e+00	 1.3920373e-01	  1.1679438e+00 	 2.1787214e-01


.. parsed-literal::

     121	 1.3292251e+00	 1.4193854e-01	 1.3811247e+00	 1.3926561e-01	  1.1727123e+00 	 2.0694709e-01
     122	 1.3321866e+00	 1.4147699e-01	 1.3840971e+00	 1.3893466e-01	  1.1796631e+00 	 1.7801523e-01


.. parsed-literal::

     123	 1.3350609e+00	 1.4121823e-01	 1.3869817e+00	 1.3891094e-01	  1.1814671e+00 	 2.0313311e-01


.. parsed-literal::

     124	 1.3375596e+00	 1.4080151e-01	 1.3895380e+00	 1.3865857e-01	[ 1.1826446e+00]	 2.2276735e-01


.. parsed-literal::

     125	 1.3395326e+00	 1.4058255e-01	 1.3915288e+00	 1.3845587e-01	[ 1.1829428e+00]	 2.1468639e-01
     126	 1.3438184e+00	 1.3998865e-01	 1.3959511e+00	 1.3776309e-01	[ 1.1887566e+00]	 1.9921279e-01


.. parsed-literal::

     127	 1.3462478e+00	 1.3952847e-01	 1.3985500e+00	 1.3759416e-01	  1.1882496e+00 	 3.1364536e-01


.. parsed-literal::

     128	 1.3494080e+00	 1.3918801e-01	 1.4017613e+00	 1.3710919e-01	[ 1.1935472e+00]	 2.0766592e-01


.. parsed-literal::

     129	 1.3529552e+00	 1.3882818e-01	 1.4053035e+00	 1.3659991e-01	[ 1.2010009e+00]	 2.1468520e-01


.. parsed-literal::

     130	 1.3557373e+00	 1.3849963e-01	 1.4081317e+00	 1.3647752e-01	  1.1993335e+00 	 2.0949674e-01


.. parsed-literal::

     131	 1.3584586e+00	 1.3835599e-01	 1.4107549e+00	 1.3643620e-01	[ 1.2034458e+00]	 2.0746398e-01
     132	 1.3606564e+00	 1.3821902e-01	 1.4129179e+00	 1.3643693e-01	[ 1.2049536e+00]	 1.8264627e-01


.. parsed-literal::

     133	 1.3638293e+00	 1.3806132e-01	 1.4161237e+00	 1.3654732e-01	  1.2018946e+00 	 2.0205331e-01


.. parsed-literal::

     134	 1.3671553e+00	 1.3770511e-01	 1.4196824e+00	 1.3608613e-01	  1.1998270e+00 	 2.0170164e-01
     135	 1.3703083e+00	 1.3753464e-01	 1.4229396e+00	 1.3595134e-01	  1.1952324e+00 	 2.0382380e-01


.. parsed-literal::

     136	 1.3738883e+00	 1.3750271e-01	 1.4267815e+00	 1.3553143e-01	  1.1871702e+00 	 2.0852280e-01


.. parsed-literal::

     137	 1.3757949e+00	 1.3739560e-01	 1.4287964e+00	 1.3525940e-01	  1.1861562e+00 	 2.1865821e-01
     138	 1.3774557e+00	 1.3723713e-01	 1.4304129e+00	 1.3506584e-01	  1.1875091e+00 	 1.6802287e-01


.. parsed-literal::

     139	 1.3805616e+00	 1.3695584e-01	 1.4334806e+00	 1.3468711e-01	  1.1900646e+00 	 2.1103191e-01
     140	 1.3827248e+00	 1.3673691e-01	 1.4356914e+00	 1.3440158e-01	  1.1894080e+00 	 2.0179033e-01


.. parsed-literal::

     141	 1.3842417e+00	 1.3670817e-01	 1.4375351e+00	 1.3427522e-01	  1.1815037e+00 	 1.8685031e-01


.. parsed-literal::

     142	 1.3889050e+00	 1.3624250e-01	 1.4420997e+00	 1.3374416e-01	  1.1780019e+00 	 2.1057630e-01


.. parsed-literal::

     143	 1.3905630e+00	 1.3612621e-01	 1.4436926e+00	 1.3360087e-01	  1.1778580e+00 	 2.0984983e-01
     144	 1.3932000e+00	 1.3582334e-01	 1.4463726e+00	 1.3321818e-01	  1.1721876e+00 	 1.8330169e-01


.. parsed-literal::

     145	 1.3964517e+00	 1.3559186e-01	 1.4496529e+00	 1.3275919e-01	  1.1721581e+00 	 2.0800233e-01
     146	 1.3994907e+00	 1.3491591e-01	 1.4528483e+00	 1.3194005e-01	  1.1713850e+00 	 1.9154215e-01


.. parsed-literal::

     147	 1.4023535e+00	 1.3490739e-01	 1.4556176e+00	 1.3182993e-01	  1.1774406e+00 	 2.0441842e-01


.. parsed-literal::

     148	 1.4047570e+00	 1.3475518e-01	 1.4580118e+00	 1.3164543e-01	  1.1813938e+00 	 2.1137786e-01


.. parsed-literal::

     149	 1.4069892e+00	 1.3459764e-01	 1.4602765e+00	 1.3129217e-01	  1.1892463e+00 	 2.1472669e-01


.. parsed-literal::

     150	 1.4098514e+00	 1.3427204e-01	 1.4631874e+00	 1.3097644e-01	  1.1892558e+00 	 2.1041203e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.28 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fd500d41150>



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
    CPU times: user 1.75 s, sys: 39.9 ms, total: 1.79 s
    Wall time: 586 ms


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

