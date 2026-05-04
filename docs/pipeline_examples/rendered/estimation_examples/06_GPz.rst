GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`GPz.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/estimation_examples/GPz.ipynb>`__
in the ``interactive_examples/estimation_examples/`` folder for a
version of this notebook in interactive mode.

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
    import tables_io
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = tables_io.read(trainFile)
    test_data = tables_io.read(testFile)

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
       1	-3.5267830e-01	 3.2339610e-01	-3.4255688e-01	 3.1244593e-01	[-3.2279444e-01]	 4.8402715e-01


.. parsed-literal::

       2	-2.8549488e-01	 3.1446320e-01	-2.6268074e-01	 3.0389314e-01	[-2.3161192e-01]	 2.3483181e-01


.. parsed-literal::

       3	-2.4019056e-01	 2.9333397e-01	-1.9616371e-01	 2.8452430e-01	[-1.6381832e-01]	 2.9203391e-01


.. parsed-literal::

       4	-2.0837752e-01	 2.7723049e-01	-1.5972299e-01	 2.7238798e-01	[-1.3570738e-01]	 3.2796335e-01
       5	-1.4137460e-01	 2.6057666e-01	-1.0161971e-01	 2.5584320e-01	[-7.6688604e-02]	 1.7995501e-01


.. parsed-literal::

       6	-7.7284719e-02	 2.5352041e-01	-4.7359530e-02	 2.4838994e-01	[-2.5972555e-02]	 1.9756985e-01


.. parsed-literal::

       7	-5.6735904e-02	 2.5008605e-01	-3.2529625e-02	 2.4429167e-01	[-9.6492476e-03]	 2.2150993e-01


.. parsed-literal::

       8	-4.1404106e-02	 2.4728430e-01	-2.2217866e-02	 2.4291917e-01	[-3.2353993e-03]	 2.0921659e-01


.. parsed-literal::

       9	-3.3559048e-02	 2.4589163e-01	-1.5901781e-02	 2.4156436e-01	[ 1.1982221e-03]	 2.1748424e-01


.. parsed-literal::

      10	-2.4812217e-02	 2.4423008e-01	-8.4627628e-03	 2.3979948e-01	[ 1.0230578e-02]	 2.1571064e-01


.. parsed-literal::

      11	-1.6774110e-02	 2.4278328e-01	-1.7612730e-03	 2.3960959e-01	[ 1.2533906e-02]	 2.1341586e-01


.. parsed-literal::

      12	-1.2001698e-02	 2.4192171e-01	 2.7655748e-03	 2.3902575e-01	[ 1.6228912e-02]	 2.0852685e-01


.. parsed-literal::

      13	-7.8823779e-03	 2.4106343e-01	 7.1361639e-03	 2.3807358e-01	[ 2.0646665e-02]	 2.2351289e-01


.. parsed-literal::

      14	 1.6011485e-01	 2.2237206e-01	 1.8614942e-01	 2.2387763e-01	[ 1.7890548e-01]	 3.4213972e-01


.. parsed-literal::

      15	 1.8722325e-01	 2.2186157e-01	 2.1482243e-01	 2.2431555e-01	[ 2.0013247e-01]	 2.1234798e-01


.. parsed-literal::

      16	 2.6681415e-01	 2.1439428e-01	 2.9599246e-01	 2.1846931e-01	[ 2.7594321e-01]	 2.1752167e-01


.. parsed-literal::

      17	 3.5911623e-01	 2.0529496e-01	 3.9236677e-01	 2.1338251e-01	[ 3.5621680e-01]	 2.1126676e-01


.. parsed-literal::

      18	 4.2079400e-01	 2.0012086e-01	 4.5417465e-01	 2.0706914e-01	[ 4.1642974e-01]	 2.2674870e-01


.. parsed-literal::

      19	 5.1187575e-01	 1.9577098e-01	 5.4704444e-01	 1.9987812e-01	[ 5.2251005e-01]	 2.1113539e-01
      20	 6.1175323e-01	 1.9528654e-01	 6.5139121e-01	 1.9978246e-01	[ 6.3569657e-01]	 1.9998431e-01


.. parsed-literal::

      21	 6.4275064e-01	 1.9349497e-01	 6.8620832e-01	 1.9323057e-01	[ 6.9588915e-01]	 2.0058393e-01


.. parsed-literal::

      22	 6.9365146e-01	 1.8913967e-01	 7.3450345e-01	 1.9108176e-01	[ 7.3115066e-01]	 2.0844817e-01
      23	 7.2181144e-01	 1.8989900e-01	 7.6198236e-01	 1.9238231e-01	[ 7.5179843e-01]	 1.9480133e-01


.. parsed-literal::

      24	 7.6103816e-01	 1.8854760e-01	 8.0012546e-01	 1.9108592e-01	[ 7.8264628e-01]	 3.2823896e-01


.. parsed-literal::

      25	 7.7039135e-01	 1.9060619e-01	 8.1012503e-01	 1.9384131e-01	  7.8063778e-01 	 2.1272635e-01


.. parsed-literal::

      26	 8.1954845e-01	 1.8386836e-01	 8.6056717e-01	 1.8745900e-01	[ 8.3811995e-01]	 2.0404220e-01
      27	 8.4224068e-01	 1.8157259e-01	 8.8384969e-01	 1.8367546e-01	[ 8.6676944e-01]	 1.9421530e-01


.. parsed-literal::

      28	 8.6719065e-01	 1.7898943e-01	 9.1015672e-01	 1.8068619e-01	[ 8.9448184e-01]	 2.1017909e-01


.. parsed-literal::

      29	 8.9425513e-01	 1.7622398e-01	 9.3799382e-01	 1.7810601e-01	[ 9.2356692e-01]	 2.1305442e-01


.. parsed-literal::

      30	 9.1295179e-01	 1.7962188e-01	 9.5762290e-01	 1.7810729e-01	[ 9.3729083e-01]	 2.1223378e-01


.. parsed-literal::

      31	 9.2776503e-01	 1.7545980e-01	 9.7289597e-01	 1.7748592e-01	[ 9.5766569e-01]	 2.0415568e-01


.. parsed-literal::

      32	 9.3700595e-01	 1.7396838e-01	 9.8190482e-01	 1.7651640e-01	[ 9.6837759e-01]	 2.1139789e-01


.. parsed-literal::

      33	 9.5004556e-01	 1.7331231e-01	 9.9518244e-01	 1.7676563e-01	[ 9.7877556e-01]	 2.1410823e-01


.. parsed-literal::

      34	 9.6891515e-01	 1.7296340e-01	 1.0146826e+00	 1.7721832e-01	[ 9.9433648e-01]	 2.1323133e-01


.. parsed-literal::

      35	 9.8422989e-01	 1.6994150e-01	 1.0311668e+00	 1.7505832e-01	[ 1.0023521e+00]	 2.1481466e-01
      36	 1.0023803e+00	 1.6752965e-01	 1.0491754e+00	 1.7347073e-01	[ 1.0214290e+00]	 2.0006609e-01


.. parsed-literal::

      37	 1.0104542e+00	 1.6609544e-01	 1.0572154e+00	 1.7142315e-01	[ 1.0333903e+00]	 2.1053672e-01


.. parsed-literal::

      38	 1.0210952e+00	 1.6418546e-01	 1.0684044e+00	 1.6880207e-01	[ 1.0454076e+00]	 2.2073054e-01


.. parsed-literal::

      39	 1.0330067e+00	 1.6283790e-01	 1.0810104e+00	 1.6671194e-01	[ 1.0585003e+00]	 2.1456337e-01


.. parsed-literal::

      40	 1.0433923e+00	 1.6206077e-01	 1.0917810e+00	 1.6562862e-01	[ 1.0651476e+00]	 2.1164989e-01


.. parsed-literal::

      41	 1.0546766e+00	 1.6167171e-01	 1.1034842e+00	 1.6501184e-01	[ 1.0740305e+00]	 2.1991730e-01


.. parsed-literal::

      42	 1.0672511e+00	 1.6111505e-01	 1.1166851e+00	 1.6501356e-01	[ 1.0803080e+00]	 2.2119927e-01


.. parsed-literal::

      43	 1.0778598e+00	 1.6171309e-01	 1.1275794e+00	 1.6442078e-01	[ 1.0932738e+00]	 2.1477413e-01


.. parsed-literal::

      44	 1.0885907e+00	 1.6031643e-01	 1.1381735e+00	 1.6438450e-01	[ 1.1031362e+00]	 2.1951747e-01


.. parsed-literal::

      45	 1.1021754e+00	 1.5892857e-01	 1.1521502e+00	 1.6513480e-01	[ 1.1111700e+00]	 2.2439432e-01


.. parsed-literal::

      46	 1.1122031e+00	 1.5882540e-01	 1.1620807e+00	 1.6579699e-01	[ 1.1156870e+00]	 2.0450163e-01


.. parsed-literal::

      47	 1.1247694e+00	 1.5867539e-01	 1.1748148e+00	 1.6669023e-01	[ 1.1160775e+00]	 2.1872568e-01


.. parsed-literal::

      48	 1.1325573e+00	 1.6026183e-01	 1.1832651e+00	 1.6781454e-01	  1.1150324e+00 	 2.1988797e-01


.. parsed-literal::

      49	 1.1435929e+00	 1.5807062e-01	 1.1942242e+00	 1.6649728e-01	[ 1.1261288e+00]	 2.1295881e-01


.. parsed-literal::

      50	 1.1528437e+00	 1.5608927e-01	 1.2036971e+00	 1.6547283e-01	[ 1.1355204e+00]	 2.1303844e-01


.. parsed-literal::

      51	 1.1641914e+00	 1.5502032e-01	 1.2152527e+00	 1.6539887e-01	[ 1.1465127e+00]	 2.2297168e-01


.. parsed-literal::

      52	 1.1773607e+00	 1.5303256e-01	 1.2294690e+00	 1.6361633e-01	[ 1.1684368e+00]	 2.1061110e-01


.. parsed-literal::

      53	 1.1890115e+00	 1.5360858e-01	 1.2408828e+00	 1.6503264e-01	[ 1.1703495e+00]	 2.1450210e-01


.. parsed-literal::

      54	 1.1944847e+00	 1.5369158e-01	 1.2461854e+00	 1.6499510e-01	[ 1.1736532e+00]	 2.1591401e-01


.. parsed-literal::

      55	 1.2038593e+00	 1.5363434e-01	 1.2556968e+00	 1.6457391e-01	[ 1.1808005e+00]	 2.1015716e-01


.. parsed-literal::

      56	 1.2150645e+00	 1.5339866e-01	 1.2673414e+00	 1.6485535e-01	[ 1.1845212e+00]	 2.0759416e-01


.. parsed-literal::

      57	 1.2248254e+00	 1.5166043e-01	 1.2777535e+00	 1.6285518e-01	[ 1.1942936e+00]	 2.1185899e-01
      58	 1.2338863e+00	 1.5081003e-01	 1.2866412e+00	 1.6247073e-01	[ 1.2031117e+00]	 1.7660046e-01


.. parsed-literal::

      59	 1.2442693e+00	 1.4928593e-01	 1.2973721e+00	 1.6143502e-01	[ 1.2066256e+00]	 2.1923256e-01


.. parsed-literal::

      60	 1.2517237e+00	 1.4785826e-01	 1.3049417e+00	 1.5981810e-01	[ 1.2145699e+00]	 2.1202326e-01


.. parsed-literal::

      61	 1.2602282e+00	 1.4725820e-01	 1.3135060e+00	 1.5925246e-01	[ 1.2154721e+00]	 2.1777439e-01


.. parsed-literal::

      62	 1.2701571e+00	 1.4681015e-01	 1.3238300e+00	 1.5893393e-01	  1.2131975e+00 	 2.1401644e-01


.. parsed-literal::

      63	 1.2796795e+00	 1.4639775e-01	 1.3335872e+00	 1.5838167e-01	  1.2113371e+00 	 2.0697737e-01


.. parsed-literal::

      64	 1.2883291e+00	 1.4593879e-01	 1.3420306e+00	 1.5815531e-01	[ 1.2155418e+00]	 2.1535301e-01
      65	 1.2940258e+00	 1.4565231e-01	 1.3477133e+00	 1.5801354e-01	[ 1.2189674e+00]	 2.0012593e-01


.. parsed-literal::

      66	 1.3030344e+00	 1.4422654e-01	 1.3574618e+00	 1.5868323e-01	  1.2018805e+00 	 2.0259404e-01


.. parsed-literal::

      67	 1.3108214e+00	 1.4328352e-01	 1.3647994e+00	 1.5831299e-01	  1.2119722e+00 	 2.1815634e-01


.. parsed-literal::

      68	 1.3152471e+00	 1.4278674e-01	 1.3692681e+00	 1.5839040e-01	  1.2148665e+00 	 2.1290994e-01


.. parsed-literal::

      69	 1.3251550e+00	 1.4171562e-01	 1.3796078e+00	 1.5885860e-01	  1.2154947e+00 	 2.1542096e-01
      70	 1.3306769e+00	 1.4103516e-01	 1.3856072e+00	 1.5926977e-01	[ 1.2190033e+00]	 1.8673396e-01


.. parsed-literal::

      71	 1.3369556e+00	 1.4050059e-01	 1.3917534e+00	 1.5877378e-01	[ 1.2268809e+00]	 1.8578458e-01


.. parsed-literal::

      72	 1.3434059e+00	 1.4025543e-01	 1.3982411e+00	 1.5815016e-01	[ 1.2311341e+00]	 2.1228576e-01


.. parsed-literal::

      73	 1.3488027e+00	 1.3966152e-01	 1.4037483e+00	 1.5778781e-01	[ 1.2379396e+00]	 2.1324015e-01


.. parsed-literal::

      74	 1.3573815e+00	 1.3899994e-01	 1.4126028e+00	 1.5766742e-01	[ 1.2422688e+00]	 2.1377993e-01


.. parsed-literal::

      75	 1.3620373e+00	 1.3893335e-01	 1.4174678e+00	 1.5868878e-01	  1.2351703e+00 	 2.1061945e-01


.. parsed-literal::

      76	 1.3670069e+00	 1.3887057e-01	 1.4221817e+00	 1.5859798e-01	[ 1.2476632e+00]	 2.0642281e-01


.. parsed-literal::

      77	 1.3711417e+00	 1.3893777e-01	 1.4262827e+00	 1.5890607e-01	[ 1.2526209e+00]	 2.1649957e-01


.. parsed-literal::

      78	 1.3757911e+00	 1.3885424e-01	 1.4310337e+00	 1.5872743e-01	[ 1.2587490e+00]	 2.1549129e-01


.. parsed-literal::

      79	 1.3804730e+00	 1.3854854e-01	 1.4361073e+00	 1.5818349e-01	[ 1.2626196e+00]	 2.1702337e-01


.. parsed-literal::

      80	 1.3859437e+00	 1.3843459e-01	 1.4416639e+00	 1.5769133e-01	[ 1.2708690e+00]	 2.1643567e-01


.. parsed-literal::

      81	 1.3895911e+00	 1.3811470e-01	 1.4452377e+00	 1.5688567e-01	[ 1.2787291e+00]	 2.0660329e-01


.. parsed-literal::

      82	 1.3949896e+00	 1.3767650e-01	 1.4509046e+00	 1.5633825e-01	[ 1.2825966e+00]	 2.2004128e-01


.. parsed-literal::

      83	 1.4005854e+00	 1.3695582e-01	 1.4567455e+00	 1.5555279e-01	[ 1.2895606e+00]	 2.1086764e-01


.. parsed-literal::

      84	 1.4054532e+00	 1.3663283e-01	 1.4614440e+00	 1.5553922e-01	[ 1.2934001e+00]	 2.1753454e-01
      85	 1.4097046e+00	 1.3637503e-01	 1.4657027e+00	 1.5570757e-01	[ 1.2947991e+00]	 2.0308876e-01


.. parsed-literal::

      86	 1.4139975e+00	 1.3588378e-01	 1.4700589e+00	 1.5560600e-01	  1.2921327e+00 	 2.2174597e-01


.. parsed-literal::

      87	 1.4181059e+00	 1.3520451e-01	 1.4743326e+00	 1.5537309e-01	  1.2866841e+00 	 2.1404338e-01


.. parsed-literal::

      88	 1.4222857e+00	 1.3479325e-01	 1.4784346e+00	 1.5494902e-01	  1.2872909e+00 	 2.2087574e-01


.. parsed-literal::

      89	 1.4259152e+00	 1.3436644e-01	 1.4821197e+00	 1.5442139e-01	  1.2884878e+00 	 2.1738219e-01


.. parsed-literal::

      90	 1.4311247e+00	 1.3370124e-01	 1.4874327e+00	 1.5394305e-01	  1.2888762e+00 	 2.1300912e-01


.. parsed-literal::

      91	 1.4344955e+00	 1.3270179e-01	 1.4911366e+00	 1.5349207e-01	  1.2762198e+00 	 2.1876025e-01


.. parsed-literal::

      92	 1.4396650e+00	 1.3255675e-01	 1.4961672e+00	 1.5343747e-01	  1.2874247e+00 	 2.1384335e-01


.. parsed-literal::

      93	 1.4421052e+00	 1.3251785e-01	 1.4985840e+00	 1.5345044e-01	  1.2879811e+00 	 2.1540523e-01


.. parsed-literal::

      94	 1.4449983e+00	 1.3225772e-01	 1.5016837e+00	 1.5334852e-01	  1.2874587e+00 	 2.1228170e-01


.. parsed-literal::

      95	 1.4490275e+00	 1.3185663e-01	 1.5059501e+00	 1.5281838e-01	  1.2861792e+00 	 2.1775460e-01


.. parsed-literal::

      96	 1.4532898e+00	 1.3147199e-01	 1.5103796e+00	 1.5238240e-01	  1.2881694e+00 	 2.1421552e-01


.. parsed-literal::

      97	 1.4581483e+00	 1.3143876e-01	 1.5153031e+00	 1.5249507e-01	  1.2913564e+00 	 2.1070910e-01
      98	 1.4600510e+00	 1.3154321e-01	 1.5173573e+00	 1.5253747e-01	  1.2820195e+00 	 2.0463800e-01


.. parsed-literal::

      99	 1.4635035e+00	 1.3167752e-01	 1.5206153e+00	 1.5287800e-01	  1.2865691e+00 	 2.0954061e-01


.. parsed-literal::

     100	 1.4651264e+00	 1.3176378e-01	 1.5222111e+00	 1.5303969e-01	  1.2836468e+00 	 2.1895218e-01


.. parsed-literal::

     101	 1.4677258e+00	 1.3177061e-01	 1.5248445e+00	 1.5309620e-01	  1.2690128e+00 	 2.1732450e-01


.. parsed-literal::

     102	 1.4697227e+00	 1.3200251e-01	 1.5269146e+00	 1.5310029e-01	  1.2584263e+00 	 2.1527743e-01


.. parsed-literal::

     103	 1.4718919e+00	 1.3167203e-01	 1.5290715e+00	 1.5272946e-01	  1.2496222e+00 	 2.2030687e-01


.. parsed-literal::

     104	 1.4736334e+00	 1.3136871e-01	 1.5308674e+00	 1.5231352e-01	  1.2406441e+00 	 2.1162510e-01


.. parsed-literal::

     105	 1.4752025e+00	 1.3117648e-01	 1.5324500e+00	 1.5207792e-01	  1.2380001e+00 	 2.1548581e-01


.. parsed-literal::

     106	 1.4785291e+00	 1.3069351e-01	 1.5360488e+00	 1.5162407e-01	  1.2324938e+00 	 2.1582675e-01


.. parsed-literal::

     107	 1.4806256e+00	 1.3045977e-01	 1.5382774e+00	 1.5126321e-01	  1.2160671e+00 	 2.1334362e-01


.. parsed-literal::

     108	 1.4823960e+00	 1.3054989e-01	 1.5399388e+00	 1.5144884e-01	  1.2262148e+00 	 2.0862269e-01


.. parsed-literal::

     109	 1.4841342e+00	 1.3058303e-01	 1.5417289e+00	 1.5153813e-01	  1.2234265e+00 	 2.1700501e-01
     110	 1.4866333e+00	 1.3056682e-01	 1.5443784e+00	 1.5145923e-01	  1.2126837e+00 	 1.9815278e-01


.. parsed-literal::

     111	 1.4903112e+00	 1.3034824e-01	 1.5482632e+00	 1.5119551e-01	  1.1892421e+00 	 2.0987058e-01


.. parsed-literal::

     112	 1.4924618e+00	 1.3026096e-01	 1.5505680e+00	 1.5056391e-01	  1.1766892e+00 	 3.4770536e-01


.. parsed-literal::

     113	 1.4943716e+00	 1.2984079e-01	 1.5524667e+00	 1.5013859e-01	  1.1645229e+00 	 2.0795369e-01


.. parsed-literal::

     114	 1.4958885e+00	 1.2949972e-01	 1.5539056e+00	 1.4973003e-01	  1.1685955e+00 	 2.0727372e-01
     115	 1.4978985e+00	 1.2911504e-01	 1.5559293e+00	 1.4915259e-01	  1.1590886e+00 	 2.0328903e-01


.. parsed-literal::

     116	 1.5005487e+00	 1.2873806e-01	 1.5586452e+00	 1.4852311e-01	  1.1637996e+00 	 2.0130897e-01


.. parsed-literal::

     117	 1.5027644e+00	 1.2853985e-01	 1.5608769e+00	 1.4809620e-01	  1.1669765e+00 	 2.1709752e-01


.. parsed-literal::

     118	 1.5046549e+00	 1.2858335e-01	 1.5628026e+00	 1.4814387e-01	  1.1613288e+00 	 2.1100259e-01


.. parsed-literal::

     119	 1.5061237e+00	 1.2840927e-01	 1.5642461e+00	 1.4805932e-01	  1.1693235e+00 	 2.1837664e-01


.. parsed-literal::

     120	 1.5075558e+00	 1.2831930e-01	 1.5656019e+00	 1.4805601e-01	  1.1718463e+00 	 2.0654249e-01


.. parsed-literal::

     121	 1.5099554e+00	 1.2804124e-01	 1.5679346e+00	 1.4771671e-01	  1.1837794e+00 	 2.2137237e-01
     122	 1.5115193e+00	 1.2781166e-01	 1.5695423e+00	 1.4748473e-01	  1.1664849e+00 	 1.9807386e-01


.. parsed-literal::

     123	 1.5129622e+00	 1.2774605e-01	 1.5709736e+00	 1.4736520e-01	  1.1710697e+00 	 2.0732927e-01


.. parsed-literal::

     124	 1.5147317e+00	 1.2757537e-01	 1.5728362e+00	 1.4712532e-01	  1.1696063e+00 	 2.1217823e-01


.. parsed-literal::

     125	 1.5161748e+00	 1.2741450e-01	 1.5743579e+00	 1.4693626e-01	  1.1639225e+00 	 2.1086764e-01


.. parsed-literal::

     126	 1.5177124e+00	 1.2694061e-01	 1.5761361e+00	 1.4645528e-01	  1.1378030e+00 	 2.1219969e-01


.. parsed-literal::

     127	 1.5203218e+00	 1.2680523e-01	 1.5786475e+00	 1.4632686e-01	  1.1434623e+00 	 2.1420908e-01


.. parsed-literal::

     128	 1.5214869e+00	 1.2675442e-01	 1.5797425e+00	 1.4628283e-01	  1.1445481e+00 	 2.1378160e-01


.. parsed-literal::

     129	 1.5232608e+00	 1.2659721e-01	 1.5814656e+00	 1.4613493e-01	  1.1442752e+00 	 2.0333672e-01


.. parsed-literal::

     130	 1.5254255e+00	 1.2641839e-01	 1.5836805e+00	 1.4579473e-01	  1.1333718e+00 	 2.2055650e-01


.. parsed-literal::

     131	 1.5270787e+00	 1.2612529e-01	 1.5854706e+00	 1.4540301e-01	  1.1349314e+00 	 2.1372199e-01
     132	 1.5284895e+00	 1.2614243e-01	 1.5869045e+00	 1.4538285e-01	  1.1253532e+00 	 1.8506479e-01


.. parsed-literal::

     133	 1.5295868e+00	 1.2611348e-01	 1.5880867e+00	 1.4527493e-01	  1.1159294e+00 	 1.8226099e-01


.. parsed-literal::

     134	 1.5309314e+00	 1.2609575e-01	 1.5895524e+00	 1.4531630e-01	  1.0980289e+00 	 2.0815754e-01


.. parsed-literal::

     135	 1.5328599e+00	 1.2590478e-01	 1.5915892e+00	 1.4513541e-01	  1.0890291e+00 	 2.0483994e-01


.. parsed-literal::

     136	 1.5346334e+00	 1.2565148e-01	 1.5933714e+00	 1.4501468e-01	  1.0795933e+00 	 2.0886064e-01
     137	 1.5363210e+00	 1.2540758e-01	 1.5949903e+00	 1.4501053e-01	  1.0870710e+00 	 2.0425129e-01


.. parsed-literal::

     138	 1.5379812e+00	 1.2505659e-01	 1.5966108e+00	 1.4480845e-01	  1.0831575e+00 	 2.1120787e-01


.. parsed-literal::

     139	 1.5400583e+00	 1.2481460e-01	 1.5986845e+00	 1.4464071e-01	  1.0832600e+00 	 2.0961857e-01


.. parsed-literal::

     140	 1.5426421e+00	 1.2448416e-01	 1.6013610e+00	 1.4440653e-01	  1.0677285e+00 	 2.0925212e-01


.. parsed-literal::

     141	 1.5442394e+00	 1.2450369e-01	 1.6030221e+00	 1.4429674e-01	  1.0639904e+00 	 2.0726800e-01


.. parsed-literal::

     142	 1.5456622e+00	 1.2448985e-01	 1.6043826e+00	 1.4431186e-01	  1.0563800e+00 	 2.0976496e-01


.. parsed-literal::

     143	 1.5469864e+00	 1.2438981e-01	 1.6056767e+00	 1.4427209e-01	  1.0450012e+00 	 2.0422220e-01


.. parsed-literal::

     144	 1.5481866e+00	 1.2425653e-01	 1.6068478e+00	 1.4421237e-01	  1.0312923e+00 	 2.0262957e-01
     145	 1.5492700e+00	 1.2399277e-01	 1.6079081e+00	 1.4405620e-01	  1.0290390e+00 	 2.0526028e-01


.. parsed-literal::

     146	 1.5506490e+00	 1.2399605e-01	 1.6092110e+00	 1.4405242e-01	  1.0282089e+00 	 2.1037340e-01


.. parsed-literal::

     147	 1.5514963e+00	 1.2399125e-01	 1.6100177e+00	 1.4399193e-01	  1.0355068e+00 	 2.1466303e-01


.. parsed-literal::

     148	 1.5523836e+00	 1.2393092e-01	 1.6108903e+00	 1.4381968e-01	  1.0425427e+00 	 2.1550226e-01


.. parsed-literal::

     149	 1.5538697e+00	 1.2388631e-01	 1.6124149e+00	 1.4356076e-01	  1.0525040e+00 	 2.1040606e-01


.. parsed-literal::

     150	 1.5551987e+00	 1.2361177e-01	 1.6138394e+00	 1.4286444e-01	  1.0606552e+00 	 2.1798682e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 9s, sys: 1.06 s, total: 2min 10s
    Wall time: 32.8 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7efce0c2b190>



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

    Inserting handle into data store.  input: None, gpz_run
    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 20,449
    Process 0 estimating GPz PZ PDF for rows 0 - 20,449


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run
    CPU times: user 951 ms, sys: 37 ms, total: 988 ms
    Wall time: 371 ms


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




.. image:: 06_GPz_files/06_GPz_15_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: 06_GPz_files/06_GPz_18_1.png

