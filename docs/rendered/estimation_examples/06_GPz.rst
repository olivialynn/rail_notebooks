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
       1	-3.4119941e-01	 3.1980741e-01	-3.3150611e-01	 3.2402681e-01	[-3.3995398e-01]	 4.7035480e-01


.. parsed-literal::

       2	-2.6890358e-01	 3.0853707e-01	-2.4435631e-01	 3.1372978e-01	[-2.6080296e-01]	 2.3438501e-01


.. parsed-literal::

       3	-2.2562500e-01	 2.8889208e-01	-1.8410777e-01	 2.9355483e-01	[-2.0529486e-01]	 2.8840399e-01
       4	-1.9114029e-01	 2.6451977e-01	-1.4990295e-01	 2.6789193e-01	[-1.7368202e-01]	 1.8572354e-01


.. parsed-literal::

       5	-1.0177854e-01	 2.5567690e-01	-6.6220138e-02	 2.6360125e-01	[-1.0413113e-01]	 2.0816255e-01
       6	-6.4798799e-02	 2.4988381e-01	-3.3020935e-02	 2.5759155e-01	[-6.3078355e-02]	 2.0153260e-01


.. parsed-literal::

       7	-4.6838321e-02	 2.4716257e-01	-2.1687817e-02	 2.5589681e-01	[-5.6916893e-02]	 2.1679616e-01


.. parsed-literal::

       8	-3.2620087e-02	 2.4478759e-01	-1.1761683e-02	 2.5459101e-01	[-5.1543794e-02]	 2.0575428e-01


.. parsed-literal::

       9	-1.7880822e-02	 2.4207139e-01	-1.8966718e-04	 2.5323989e-01	[-4.5800712e-02]	 2.1109390e-01


.. parsed-literal::

      10	-7.2904574e-03	 2.4021077e-01	 8.1285768e-03	 2.5264311e-01	[-4.0268008e-02]	 2.1338797e-01
      11	-1.4870064e-03	 2.3912306e-01	 1.2820807e-02	 2.5208787e-01	 -4.0957538e-02 	 1.8252206e-01


.. parsed-literal::

      12	 1.1101542e-03	 2.3865091e-01	 1.5264517e-02	 2.5152755e-01	[-3.6060240e-02]	 1.7457509e-01


.. parsed-literal::

      13	 4.3774352e-03	 2.3807718e-01	 1.8300477e-02	 2.5056758e-01	[-3.2415012e-02]	 2.1269846e-01


.. parsed-literal::

      14	 8.5489053e-03	 2.3717077e-01	 2.2827121e-02	 2.4975252e-01	[-2.7214710e-02]	 2.0516086e-01


.. parsed-literal::

      15	 6.5890067e-02	 2.2313494e-01	 8.5131716e-02	 2.3621566e-01	[ 5.5463874e-02]	 3.2028723e-01
      16	 1.4176652e-01	 2.1770648e-01	 1.6320748e-01	 2.3208531e-01	[ 1.2382663e-01]	 1.8477654e-01


.. parsed-literal::

      17	 2.0034415e-01	 2.1350036e-01	 2.3284504e-01	 2.2647717e-01	[ 1.5088739e-01]	 2.0875192e-01
      18	 2.7367681e-01	 2.1676102e-01	 3.0378640e-01	 2.2632202e-01	[ 2.6303275e-01]	 1.8922210e-01


.. parsed-literal::

      19	 3.0263439e-01	 2.1120667e-01	 3.3270974e-01	 2.2197839e-01	[ 2.9299627e-01]	 2.1327853e-01
      20	 3.3449038e-01	 2.0803981e-01	 3.6550333e-01	 2.1933975e-01	[ 3.2387775e-01]	 1.9676352e-01


.. parsed-literal::

      21	 3.9974501e-01	 2.0457636e-01	 4.3110941e-01	 2.1664154e-01	[ 3.8027503e-01]	 2.0159197e-01


.. parsed-literal::

      22	 5.1046829e-01	 2.0250154e-01	 5.4625619e-01	 2.1308666e-01	[ 4.9506662e-01]	 2.0845795e-01


.. parsed-literal::

      23	 5.6202100e-01	 2.0599315e-01	 6.0247927e-01	 2.1581470e-01	[ 5.4305993e-01]	 2.0833087e-01
      24	 6.1265987e-01	 2.0124463e-01	 6.4984206e-01	 2.1166238e-01	[ 6.0655316e-01]	 1.8830824e-01


.. parsed-literal::

      25	 6.4858278e-01	 1.9607701e-01	 6.8493353e-01	 2.0856597e-01	[ 6.2845030e-01]	 2.2436786e-01


.. parsed-literal::

      26	 6.7568517e-01	 1.9487089e-01	 7.0908416e-01	 2.0746562e-01	[ 6.4549767e-01]	 2.1855760e-01


.. parsed-literal::

      27	 7.2744707e-01	 1.9108146e-01	 7.6251686e-01	 2.0441591e-01	[ 6.8791680e-01]	 2.2021413e-01


.. parsed-literal::

      28	 7.4788165e-01	 1.8987816e-01	 7.8395229e-01	 2.0193313e-01	[ 7.2358365e-01]	 3.2381177e-01


.. parsed-literal::

      29	 7.7379267e-01	 1.9266807e-01	 8.1027972e-01	 2.0563021e-01	[ 7.4756111e-01]	 2.1180725e-01


.. parsed-literal::

      30	 7.9170062e-01	 1.9115061e-01	 8.2862311e-01	 2.0256681e-01	[ 7.7560903e-01]	 2.1007848e-01
      31	 8.0823461e-01	 1.8968730e-01	 8.4592028e-01	 2.0042981e-01	[ 7.9804578e-01]	 1.9490027e-01


.. parsed-literal::

      32	 8.3464189e-01	 1.8912513e-01	 8.7364027e-01	 2.0263279e-01	[ 8.1384868e-01]	 2.1799850e-01


.. parsed-literal::

      33	 8.6737693e-01	 1.8329179e-01	 9.0731723e-01	 1.9794442e-01	[ 8.4763666e-01]	 2.0834899e-01
      34	 8.9428136e-01	 1.8143690e-01	 9.3457998e-01	 1.9745947e-01	[ 8.7643479e-01]	 1.8946886e-01


.. parsed-literal::

      35	 9.1106141e-01	 1.7849317e-01	 9.5169438e-01	 1.9373743e-01	[ 8.9684611e-01]	 2.0848155e-01


.. parsed-literal::

      36	 9.2370842e-01	 1.7587063e-01	 9.6400442e-01	 1.9194632e-01	[ 9.0804910e-01]	 2.1129060e-01


.. parsed-literal::

      37	 9.3488582e-01	 1.7398832e-01	 9.7520178e-01	 1.9009033e-01	[ 9.2026551e-01]	 2.1939301e-01


.. parsed-literal::

      38	 9.5234034e-01	 1.7157680e-01	 9.9318170e-01	 1.8765081e-01	[ 9.3966714e-01]	 2.2230124e-01


.. parsed-literal::

      39	 9.6703156e-01	 1.7189600e-01	 1.0091578e+00	 1.8733919e-01	[ 9.5531927e-01]	 2.1476412e-01


.. parsed-literal::

      40	 9.8278758e-01	 1.7032956e-01	 1.0252733e+00	 1.8650561e-01	[ 9.7216631e-01]	 2.1423459e-01


.. parsed-literal::

      41	 9.9073049e-01	 1.7004323e-01	 1.0333282e+00	 1.8654376e-01	[ 9.7980399e-01]	 2.1434522e-01


.. parsed-literal::

      42	 1.0018852e+00	 1.7000238e-01	 1.0449689e+00	 1.8743713e-01	[ 9.8881315e-01]	 2.1598625e-01


.. parsed-literal::

      43	 1.0126429e+00	 1.7100305e-01	 1.0565483e+00	 1.9081922e-01	[ 9.9000435e-01]	 2.0908427e-01


.. parsed-literal::

      44	 1.0214283e+00	 1.7022581e-01	 1.0654532e+00	 1.9096832e-01	[ 9.9807528e-01]	 2.1119452e-01


.. parsed-literal::

      45	 1.0266733e+00	 1.6938443e-01	 1.0704341e+00	 1.9010408e-01	[ 1.0025939e+00]	 2.0552850e-01


.. parsed-literal::

      46	 1.0371632e+00	 1.6825463e-01	 1.0808695e+00	 1.8914588e-01	[ 1.0161974e+00]	 2.1200252e-01
      47	 1.0452794e+00	 1.6653430e-01	 1.0890287e+00	 1.8653480e-01	[ 1.0272569e+00]	 1.8068695e-01


.. parsed-literal::

      48	 1.0530030e+00	 1.6597983e-01	 1.0968617e+00	 1.8639170e-01	[ 1.0377252e+00]	 2.1008921e-01
      49	 1.0609562e+00	 1.6527521e-01	 1.1049965e+00	 1.8609393e-01	[ 1.0454517e+00]	 1.8317652e-01


.. parsed-literal::

      50	 1.0695554e+00	 1.6493169e-01	 1.1140906e+00	 1.8592642e-01	[ 1.0521177e+00]	 1.9021869e-01


.. parsed-literal::

      51	 1.0770379e+00	 1.6437006e-01	 1.1222396e+00	 1.8479032e-01	[ 1.0610552e+00]	 2.0998788e-01


.. parsed-literal::

      52	 1.0833223e+00	 1.6374754e-01	 1.1284012e+00	 1.8405179e-01	[ 1.0623148e+00]	 2.1150827e-01


.. parsed-literal::

      53	 1.0890356e+00	 1.6341457e-01	 1.1340331e+00	 1.8365351e-01	[ 1.0654489e+00]	 2.0728207e-01


.. parsed-literal::

      54	 1.0981487e+00	 1.6243529e-01	 1.1430743e+00	 1.8245682e-01	[ 1.0733638e+00]	 2.1633148e-01


.. parsed-literal::

      55	 1.1101139e+00	 1.5978793e-01	 1.1552914e+00	 1.7923233e-01	[ 1.0919084e+00]	 2.1213627e-01


.. parsed-literal::

      56	 1.1128073e+00	 1.5726852e-01	 1.1584442e+00	 1.7758459e-01	[ 1.0932432e+00]	 2.0875549e-01


.. parsed-literal::

      57	 1.1232599e+00	 1.5649697e-01	 1.1686371e+00	 1.7644934e-01	[ 1.1038923e+00]	 2.0976114e-01


.. parsed-literal::

      58	 1.1269285e+00	 1.5586182e-01	 1.1724863e+00	 1.7572217e-01	[ 1.1065425e+00]	 2.1081066e-01


.. parsed-literal::

      59	 1.1333926e+00	 1.5458646e-01	 1.1792774e+00	 1.7401879e-01	[ 1.1087195e+00]	 2.1220183e-01


.. parsed-literal::

      60	 1.1400686e+00	 1.5307009e-01	 1.1862914e+00	 1.7267744e-01	[ 1.1109383e+00]	 2.0366812e-01
      61	 1.1468351e+00	 1.5144153e-01	 1.1932237e+00	 1.7050510e-01	[ 1.1132041e+00]	 2.0301867e-01


.. parsed-literal::

      62	 1.1517158e+00	 1.5060137e-01	 1.1981078e+00	 1.6955806e-01	[ 1.1158497e+00]	 2.1217585e-01
      63	 1.1596133e+00	 1.4860553e-01	 1.2062179e+00	 1.6797909e-01	[ 1.1172174e+00]	 1.7533088e-01


.. parsed-literal::

      64	 1.1635498e+00	 1.4801225e-01	 1.2105578e+00	 1.6771158e-01	  1.1161274e+00 	 2.1013379e-01


.. parsed-literal::

      65	 1.1685852e+00	 1.4741205e-01	 1.2154936e+00	 1.6748310e-01	[ 1.1209914e+00]	 2.0441699e-01


.. parsed-literal::

      66	 1.1736135e+00	 1.4660248e-01	 1.2207167e+00	 1.6728888e-01	[ 1.1241850e+00]	 2.1077681e-01


.. parsed-literal::

      67	 1.1781414e+00	 1.4622064e-01	 1.2254082e+00	 1.6719474e-01	[ 1.1288423e+00]	 2.1427488e-01


.. parsed-literal::

      68	 1.1877112e+00	 1.4545951e-01	 1.2356072e+00	 1.6673606e-01	[ 1.1348834e+00]	 2.1647882e-01


.. parsed-literal::

      69	 1.1922415e+00	 1.4537199e-01	 1.2404096e+00	 1.6663998e-01	[ 1.1409635e+00]	 3.3554769e-01


.. parsed-literal::

      70	 1.1970389e+00	 1.4481067e-01	 1.2452840e+00	 1.6596009e-01	[ 1.1447332e+00]	 2.0266080e-01


.. parsed-literal::

      71	 1.2036383e+00	 1.4351193e-01	 1.2520405e+00	 1.6478750e-01	[ 1.1490158e+00]	 2.1297646e-01


.. parsed-literal::

      72	 1.2086419e+00	 1.4224651e-01	 1.2573269e+00	 1.6335212e-01	[ 1.1530480e+00]	 2.1018028e-01


.. parsed-literal::

      73	 1.2145290e+00	 1.4189577e-01	 1.2631445e+00	 1.6314521e-01	[ 1.1585735e+00]	 2.1451664e-01


.. parsed-literal::

      74	 1.2209935e+00	 1.4144663e-01	 1.2696749e+00	 1.6308303e-01	[ 1.1652049e+00]	 2.1879530e-01
      75	 1.2254136e+00	 1.4129803e-01	 1.2742333e+00	 1.6322072e-01	[ 1.1687796e+00]	 1.8022037e-01


.. parsed-literal::

      76	 1.2316967e+00	 1.4061771e-01	 1.2807568e+00	 1.6336330e-01	[ 1.1762936e+00]	 2.1255875e-01
      77	 1.2369357e+00	 1.4026598e-01	 1.2861174e+00	 1.6314159e-01	[ 1.1792981e+00]	 1.8149424e-01


.. parsed-literal::

      78	 1.2428780e+00	 1.3969549e-01	 1.2921613e+00	 1.6252812e-01	[ 1.1836323e+00]	 2.1058679e-01
      79	 1.2477540e+00	 1.3911291e-01	 1.2971555e+00	 1.6161248e-01	[ 1.1899687e+00]	 1.9829988e-01


.. parsed-literal::

      80	 1.2527935e+00	 1.3856421e-01	 1.3022996e+00	 1.6100186e-01	[ 1.1947581e+00]	 2.0210862e-01
      81	 1.2563776e+00	 1.3841234e-01	 1.3057644e+00	 1.6106996e-01	[ 1.2011790e+00]	 1.8590474e-01


.. parsed-literal::

      82	 1.2610605e+00	 1.3821123e-01	 1.3105414e+00	 1.6122388e-01	[ 1.2056399e+00]	 2.0208287e-01


.. parsed-literal::

      83	 1.2683855e+00	 1.3809635e-01	 1.3180526e+00	 1.6231298e-01	[ 1.2105108e+00]	 2.1622658e-01


.. parsed-literal::

      84	 1.2753689e+00	 1.3842608e-01	 1.3250881e+00	 1.6360263e-01	[ 1.2153799e+00]	 2.0470262e-01


.. parsed-literal::

      85	 1.2807884e+00	 1.3848211e-01	 1.3305662e+00	 1.6433795e-01	[ 1.2175752e+00]	 2.1126318e-01


.. parsed-literal::

      86	 1.2868082e+00	 1.3852967e-01	 1.3367850e+00	 1.6488858e-01	[ 1.2226079e+00]	 2.1249270e-01


.. parsed-literal::

      87	 1.2919676e+00	 1.3856129e-01	 1.3421462e+00	 1.6552752e-01	[ 1.2248595e+00]	 2.0437527e-01


.. parsed-literal::

      88	 1.2972233e+00	 1.3842582e-01	 1.3474253e+00	 1.6523525e-01	[ 1.2317144e+00]	 2.1366572e-01


.. parsed-literal::

      89	 1.3016850e+00	 1.3801380e-01	 1.3520013e+00	 1.6447598e-01	[ 1.2381508e+00]	 2.0527697e-01
      90	 1.3064754e+00	 1.3739932e-01	 1.3570596e+00	 1.6400975e-01	[ 1.2404246e+00]	 2.0686126e-01


.. parsed-literal::

      91	 1.3105504e+00	 1.3718099e-01	 1.3613390e+00	 1.6388464e-01	[ 1.2412867e+00]	 2.1248579e-01
      92	 1.3143551e+00	 1.3694436e-01	 1.3653663e+00	 1.6402383e-01	[ 1.2431226e+00]	 1.9168663e-01


.. parsed-literal::

      93	 1.3197521e+00	 1.3703029e-01	 1.3710574e+00	 1.6466243e-01	[ 1.2452372e+00]	 1.9977784e-01


.. parsed-literal::

      94	 1.3251876e+00	 1.3699217e-01	 1.3766629e+00	 1.6474878e-01	[ 1.2468221e+00]	 2.1487498e-01


.. parsed-literal::

      95	 1.3297838e+00	 1.3705126e-01	 1.3813230e+00	 1.6484003e-01	[ 1.2509387e+00]	 2.1538186e-01


.. parsed-literal::

      96	 1.3354052e+00	 1.3706237e-01	 1.3871639e+00	 1.6471667e-01	[ 1.2560016e+00]	 2.1040154e-01


.. parsed-literal::

      97	 1.3389999e+00	 1.3724560e-01	 1.3909488e+00	 1.6509365e-01	[ 1.2593555e+00]	 2.1239638e-01


.. parsed-literal::

      98	 1.3428603e+00	 1.3684394e-01	 1.3946520e+00	 1.6428117e-01	[ 1.2668964e+00]	 2.1131492e-01


.. parsed-literal::

      99	 1.3463788e+00	 1.3654053e-01	 1.3982210e+00	 1.6372126e-01	[ 1.2700967e+00]	 2.1452403e-01


.. parsed-literal::

     100	 1.3499119e+00	 1.3633078e-01	 1.4018577e+00	 1.6321487e-01	[ 1.2708293e+00]	 2.1025729e-01


.. parsed-literal::

     101	 1.3535445e+00	 1.3621695e-01	 1.4057364e+00	 1.6280503e-01	  1.2707290e+00 	 2.1583724e-01


.. parsed-literal::

     102	 1.3574527e+00	 1.3611427e-01	 1.4095633e+00	 1.6260905e-01	[ 1.2737537e+00]	 2.0488000e-01


.. parsed-literal::

     103	 1.3600731e+00	 1.3605030e-01	 1.4121721e+00	 1.6261570e-01	[ 1.2757512e+00]	 2.1386957e-01
     104	 1.3635176e+00	 1.3592221e-01	 1.4158207e+00	 1.6247860e-01	[ 1.2779867e+00]	 1.7704868e-01


.. parsed-literal::

     105	 1.3671762e+00	 1.3575219e-01	 1.4196299e+00	 1.6247844e-01	[ 1.2815480e+00]	 2.1304345e-01
     106	 1.3703303e+00	 1.3550911e-01	 1.4228433e+00	 1.6223892e-01	[ 1.2839834e+00]	 2.0043421e-01


.. parsed-literal::

     107	 1.3743731e+00	 1.3526541e-01	 1.4271518e+00	 1.6202121e-01	[ 1.2865863e+00]	 1.9424558e-01


.. parsed-literal::

     108	 1.3766749e+00	 1.3499735e-01	 1.4295464e+00	 1.6180882e-01	  1.2859746e+00 	 2.1114802e-01


.. parsed-literal::

     109	 1.3794619e+00	 1.3500619e-01	 1.4323304e+00	 1.6188089e-01	[ 1.2887475e+00]	 2.1988964e-01
     110	 1.3842297e+00	 1.3511847e-01	 1.4372379e+00	 1.6210206e-01	[ 1.2937537e+00]	 1.8947411e-01


.. parsed-literal::

     111	 1.3865137e+00	 1.3521072e-01	 1.4396114e+00	 1.6240717e-01	  1.2925042e+00 	 2.1369648e-01
     112	 1.3894881e+00	 1.3522860e-01	 1.4426490e+00	 1.6240688e-01	[ 1.2949583e+00]	 1.8102121e-01


.. parsed-literal::

     113	 1.3938356e+00	 1.3493988e-01	 1.4472336e+00	 1.6189054e-01	[ 1.2976178e+00]	 1.7256379e-01
     114	 1.3964814e+00	 1.3493584e-01	 1.4499826e+00	 1.6197379e-01	[ 1.2981788e+00]	 1.9776893e-01


.. parsed-literal::

     115	 1.3986929e+00	 1.3461144e-01	 1.4520749e+00	 1.6157747e-01	[ 1.3007898e+00]	 1.8505645e-01
     116	 1.4026046e+00	 1.3414455e-01	 1.4560416e+00	 1.6109534e-01	[ 1.3028185e+00]	 2.0095420e-01


.. parsed-literal::

     117	 1.4050632e+00	 1.3380747e-01	 1.4585198e+00	 1.6075651e-01	[ 1.3054556e+00]	 2.1268225e-01
     118	 1.4079820e+00	 1.3379038e-01	 1.4614756e+00	 1.6080911e-01	[ 1.3070346e+00]	 1.7567563e-01


.. parsed-literal::

     119	 1.4112301e+00	 1.3378506e-01	 1.4647625e+00	 1.6103595e-01	[ 1.3081643e+00]	 2.2017765e-01


.. parsed-literal::

     120	 1.4141279e+00	 1.3376849e-01	 1.4678160e+00	 1.6108327e-01	[ 1.3087446e+00]	 2.1127534e-01


.. parsed-literal::

     121	 1.4171348e+00	 1.3349458e-01	 1.4708141e+00	 1.6099773e-01	[ 1.3105032e+00]	 2.1472669e-01
     122	 1.4194467e+00	 1.3319282e-01	 1.4731647e+00	 1.6072910e-01	[ 1.3127215e+00]	 1.9974852e-01


.. parsed-literal::

     123	 1.4225779e+00	 1.3279319e-01	 1.4763840e+00	 1.6028280e-01	[ 1.3173826e+00]	 2.1247530e-01


.. parsed-literal::

     124	 1.4240622e+00	 1.3264893e-01	 1.4780780e+00	 1.6055809e-01	[ 1.3174452e+00]	 2.0437074e-01
     125	 1.4270038e+00	 1.3255076e-01	 1.4808982e+00	 1.6029963e-01	[ 1.3220810e+00]	 1.9761062e-01


.. parsed-literal::

     126	 1.4290466e+00	 1.3246033e-01	 1.4829309e+00	 1.6020367e-01	[ 1.3234382e+00]	 2.1304274e-01


.. parsed-literal::

     127	 1.4314612e+00	 1.3238904e-01	 1.4854342e+00	 1.6032364e-01	[ 1.3236269e+00]	 2.0813608e-01


.. parsed-literal::

     128	 1.4338173e+00	 1.3208976e-01	 1.4880076e+00	 1.6018584e-01	  1.3182542e+00 	 2.1113253e-01


.. parsed-literal::

     129	 1.4363636e+00	 1.3196635e-01	 1.4905781e+00	 1.6019948e-01	  1.3190614e+00 	 2.1655011e-01


.. parsed-literal::

     130	 1.4387786e+00	 1.3175127e-01	 1.4930872e+00	 1.6008052e-01	  1.3192392e+00 	 2.1306419e-01


.. parsed-literal::

     131	 1.4408366e+00	 1.3153819e-01	 1.4951998e+00	 1.5985100e-01	  1.3200301e+00 	 2.0975065e-01


.. parsed-literal::

     132	 1.4443973e+00	 1.3130115e-01	 1.4989898e+00	 1.5966045e-01	  1.3174049e+00 	 2.1049523e-01


.. parsed-literal::

     133	 1.4461883e+00	 1.3098556e-01	 1.5008564e+00	 1.5927594e-01	  1.3195741e+00 	 2.1155596e-01
     134	 1.4482109e+00	 1.3108880e-01	 1.5026966e+00	 1.5934584e-01	  1.3228042e+00 	 1.7685866e-01


.. parsed-literal::

     135	 1.4496098e+00	 1.3123630e-01	 1.5040901e+00	 1.5963377e-01	  1.3216108e+00 	 1.8839383e-01


.. parsed-literal::

     136	 1.4516894e+00	 1.3122003e-01	 1.5061945e+00	 1.5967025e-01	  1.3208934e+00 	 2.0228219e-01


.. parsed-literal::

     137	 1.4542182e+00	 1.3150794e-01	 1.5087775e+00	 1.6036981e-01	  1.3186088e+00 	 2.0581961e-01


.. parsed-literal::

     138	 1.4568857e+00	 1.3130848e-01	 1.5114992e+00	 1.6030835e-01	  1.3180955e+00 	 2.1583009e-01
     139	 1.4593398e+00	 1.3113753e-01	 1.5139586e+00	 1.6023397e-01	  1.3198255e+00 	 1.7624187e-01


.. parsed-literal::

     140	 1.4611044e+00	 1.3107566e-01	 1.5157042e+00	 1.6022470e-01	  1.3214053e+00 	 2.0246840e-01
     141	 1.4628548e+00	 1.3095883e-01	 1.5174282e+00	 1.6001467e-01	  1.3233599e+00 	 2.0342469e-01


.. parsed-literal::

     142	 1.4643506e+00	 1.3095701e-01	 1.5189212e+00	 1.6006612e-01	  1.3225829e+00 	 2.1215272e-01


.. parsed-literal::

     143	 1.4660530e+00	 1.3093480e-01	 1.5206549e+00	 1.6019131e-01	  1.3189768e+00 	 2.0625043e-01


.. parsed-literal::

     144	 1.4683288e+00	 1.3071525e-01	 1.5230511e+00	 1.6011512e-01	  1.3163209e+00 	 2.0532036e-01


.. parsed-literal::

     145	 1.4703689e+00	 1.3059747e-01	 1.5251187e+00	 1.6024587e-01	  1.3109949e+00 	 2.2064543e-01


.. parsed-literal::

     146	 1.4716639e+00	 1.3054277e-01	 1.5263955e+00	 1.6020299e-01	  1.3104762e+00 	 2.0669913e-01
     147	 1.4731498e+00	 1.3042251e-01	 1.5278920e+00	 1.6010057e-01	  1.3101732e+00 	 1.7244387e-01


.. parsed-literal::

     148	 1.4747006e+00	 1.3038396e-01	 1.5295105e+00	 1.6003063e-01	  1.3029225e+00 	 2.1176815e-01


.. parsed-literal::

     149	 1.4762743e+00	 1.3025855e-01	 1.5311591e+00	 1.5989552e-01	  1.3013611e+00 	 2.1454692e-01


.. parsed-literal::

     150	 1.4777551e+00	 1.3033085e-01	 1.5327667e+00	 1.6011839e-01	  1.2981057e+00 	 2.1318841e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.2 s, total: 2min 6s
    Wall time: 31.8 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f2244724d60>



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
    CPU times: user 1.88 s, sys: 57 ms, total: 1.94 s
    Wall time: 670 ms


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

