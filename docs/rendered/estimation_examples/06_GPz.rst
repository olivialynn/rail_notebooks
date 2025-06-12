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
       1	-3.3350405e-01	 3.1717029e-01	-3.2380104e-01	 3.3395526e-01	[-3.5667279e-01]	 4.6975923e-01


.. parsed-literal::

       2	-2.6054293e-01	 3.0602160e-01	-2.3642335e-01	 3.2157105e-01	[-2.8483886e-01]	 2.3177600e-01


.. parsed-literal::

       3	-2.1561514e-01	 2.8598793e-01	-1.7437466e-01	 2.9709195e-01	[-2.2433040e-01]	 2.8735423e-01
       4	-1.8873105e-01	 2.6201436e-01	-1.5028964e-01	 2.7160749e-01	[-2.1376757e-01]	 1.7487478e-01


.. parsed-literal::

       5	-9.4204492e-02	 2.5442589e-01	-5.9093920e-02	 2.6641941e-01	[-1.1876108e-01]	 2.0879507e-01
       6	-6.0187473e-02	 2.4850839e-01	-2.7749500e-02	 2.6153149e-01	[-7.7845369e-02]	 2.0748854e-01


.. parsed-literal::

       7	-4.2459994e-02	 2.4611208e-01	-1.7476942e-02	 2.5936968e-01	[-7.1362974e-02]	 2.0361114e-01


.. parsed-literal::

       8	-2.8760870e-02	 2.4386306e-01	-8.0108412e-03	 2.5723360e-01	[-6.3197517e-02]	 2.2041535e-01


.. parsed-literal::

       9	-1.3767704e-02	 2.4114071e-01	 3.6040842e-03	 2.5490150e-01	[-5.5541699e-02]	 2.0934653e-01


.. parsed-literal::

      10	-8.7235238e-03	 2.4032647e-01	 6.8516218e-03	 2.5459617e-01	[-5.0328776e-02]	 2.0276165e-01


.. parsed-literal::

      11	-1.0110199e-03	 2.3889771e-01	 1.3762881e-02	 2.5318423e-01	[-4.9113896e-02]	 2.0718503e-01


.. parsed-literal::

      12	 1.8686519e-03	 2.3834977e-01	 1.6386268e-02	 2.5241963e-01	[-4.4376927e-02]	 2.2454596e-01


.. parsed-literal::

      13	 5.2000492e-03	 2.3773638e-01	 1.9466300e-02	 2.5162627e-01	[-4.0341190e-02]	 2.1835589e-01


.. parsed-literal::

      14	 1.1128495e-02	 2.3658649e-01	 2.5699346e-02	 2.5043489e-01	[-3.2552965e-02]	 2.1734118e-01


.. parsed-literal::

      15	 1.1388513e-01	 2.1927277e-01	 1.3562037e-01	 2.3437391e-01	[ 9.1023365e-02]	 3.3416152e-01


.. parsed-literal::

      16	 1.5085493e-01	 2.1731099e-01	 1.7385607e-01	 2.3208759e-01	[ 1.4454881e-01]	 2.2312808e-01
      17	 2.7764795e-01	 2.1329064e-01	 3.0649460e-01	 2.2409761e-01	[ 2.7462916e-01]	 1.9505405e-01


.. parsed-literal::

      18	 3.2714879e-01	 2.0932583e-01	 3.6024228e-01	 2.2050106e-01	[ 3.1026410e-01]	 2.0521712e-01


.. parsed-literal::

      19	 3.7991531e-01	 2.0587885e-01	 4.1325418e-01	 2.1725161e-01	[ 3.7059966e-01]	 2.0394135e-01


.. parsed-literal::

      20	 4.1878144e-01	 2.0420082e-01	 4.5172139e-01	 2.1630038e-01	[ 4.1492773e-01]	 2.1738839e-01


.. parsed-literal::

      21	 4.8285489e-01	 2.0040009e-01	 5.1629253e-01	 2.1267382e-01	[ 4.8314948e-01]	 2.1312690e-01


.. parsed-literal::

      22	 6.0456020e-01	 1.9789085e-01	 6.4473610e-01	 2.1011844e-01	[ 6.0215573e-01]	 2.1164465e-01


.. parsed-literal::

      23	 6.1237276e-01	 1.9926177e-01	 6.5456568e-01	 2.1272005e-01	[ 6.1793989e-01]	 2.1842885e-01


.. parsed-literal::

      24	 6.6310631e-01	 1.9488099e-01	 7.0116884e-01	 2.0772749e-01	[ 6.6878668e-01]	 2.1319699e-01


.. parsed-literal::

      25	 6.8950134e-01	 1.9383128e-01	 7.2647025e-01	 2.0581508e-01	[ 6.8898794e-01]	 2.1112347e-01


.. parsed-literal::

      26	 7.1960428e-01	 1.9362356e-01	 7.5490067e-01	 2.0941535e-01	[ 6.9776478e-01]	 2.1996951e-01


.. parsed-literal::

      27	 7.6036973e-01	 1.9524372e-01	 7.9721539e-01	 2.1065243e-01	[ 7.4138989e-01]	 2.0171452e-01


.. parsed-literal::

      28	 7.8634726e-01	 1.9563023e-01	 8.2477505e-01	 2.1177583e-01	[ 7.7648366e-01]	 2.0834684e-01


.. parsed-literal::

      29	 8.0772866e-01	 1.9979342e-01	 8.4626285e-01	 2.1735740e-01	[ 7.9600261e-01]	 2.0573807e-01
      30	 8.2662272e-01	 1.9877617e-01	 8.6639989e-01	 2.1633580e-01	[ 8.2026896e-01]	 2.0527697e-01


.. parsed-literal::

      31	 8.5451399e-01	 1.9668435e-01	 8.9600214e-01	 2.1463678e-01	[ 8.4668606e-01]	 2.0741153e-01


.. parsed-literal::

      32	 8.7839771e-01	 1.9458974e-01	 9.2173305e-01	 2.1342802e-01	[ 8.6848530e-01]	 2.1609473e-01
      33	 8.9955661e-01	 1.9193652e-01	 9.4368793e-01	 2.1044812e-01	[ 8.8583595e-01]	 1.8727016e-01


.. parsed-literal::

      34	 9.1618574e-01	 1.9222044e-01	 9.5979644e-01	 2.0829207e-01	[ 8.8989113e-01]	 2.1003604e-01


.. parsed-literal::

      35	 9.3616934e-01	 1.9094194e-01	 9.8013984e-01	 2.0909824e-01	[ 9.0453021e-01]	 2.1333623e-01


.. parsed-literal::

      36	 9.4864101e-01	 1.9006267e-01	 9.9263636e-01	 2.0792557e-01	[ 9.2046512e-01]	 2.0571089e-01


.. parsed-literal::

      37	 9.6163740e-01	 1.8992504e-01	 1.0060614e+00	 2.0685917e-01	[ 9.3471642e-01]	 2.0863795e-01
      38	 9.7465855e-01	 1.8959963e-01	 1.0196100e+00	 2.0670986e-01	[ 9.4686822e-01]	 1.9931579e-01


.. parsed-literal::

      39	 9.9087214e-01	 1.8757049e-01	 1.0366490e+00	 2.0561019e-01	[ 9.5852857e-01]	 2.0730162e-01


.. parsed-literal::

      40	 1.0056931e+00	 1.8505312e-01	 1.0521324e+00	 2.0294400e-01	[ 9.7412062e-01]	 2.1912599e-01


.. parsed-literal::

      41	 1.0203554e+00	 1.8326118e-01	 1.0671168e+00	 2.0146905e-01	[ 9.8476718e-01]	 2.1025538e-01
      42	 1.0371349e+00	 1.8010348e-01	 1.0842260e+00	 1.9842963e-01	[ 9.9777945e-01]	 1.8744397e-01


.. parsed-literal::

      43	 1.0509553e+00	 1.7749899e-01	 1.0986212e+00	 1.9728756e-01	[ 1.0021032e+00]	 2.1695280e-01


.. parsed-literal::

      44	 1.0663048e+00	 1.7389976e-01	 1.1138153e+00	 1.9310011e-01	[ 1.0186357e+00]	 2.0705938e-01
      45	 1.0771970e+00	 1.7087110e-01	 1.1246632e+00	 1.9004434e-01	[ 1.0293467e+00]	 1.8285942e-01


.. parsed-literal::

      46	 1.0943181e+00	 1.6782823e-01	 1.1420326e+00	 1.8748294e-01	[ 1.0442076e+00]	 2.0388699e-01


.. parsed-literal::

      47	 1.1107128e+00	 1.6252238e-01	 1.1590910e+00	 1.8199324e-01	[ 1.0530738e+00]	 2.0574307e-01


.. parsed-literal::

      48	 1.1284751e+00	 1.6332079e-01	 1.1767553e+00	 1.8394785e-01	[ 1.0620886e+00]	 2.0930123e-01


.. parsed-literal::

      49	 1.1381806e+00	 1.6309683e-01	 1.1867422e+00	 1.8461630e-01	[ 1.0629901e+00]	 2.0664263e-01


.. parsed-literal::

      50	 1.1591426e+00	 1.6057858e-01	 1.2080142e+00	 1.8349129e-01	[ 1.0672478e+00]	 2.1261597e-01


.. parsed-literal::

      51	 1.1644976e+00	 1.5667911e-01	 1.2167413e+00	 1.8062278e-01	  1.0186395e+00 	 2.1644855e-01


.. parsed-literal::

      52	 1.1855594e+00	 1.5591627e-01	 1.2360570e+00	 1.7924276e-01	[ 1.0685710e+00]	 2.4618173e-01


.. parsed-literal::

      53	 1.1931590e+00	 1.5513116e-01	 1.2433123e+00	 1.7822315e-01	[ 1.0850305e+00]	 2.1205521e-01


.. parsed-literal::

      54	 1.2095968e+00	 1.5213678e-01	 1.2599852e+00	 1.7453210e-01	[ 1.1097431e+00]	 2.1845388e-01
      55	 1.2226440e+00	 1.4878626e-01	 1.2735819e+00	 1.7009488e-01	[ 1.1321559e+00]	 1.9574618e-01


.. parsed-literal::

      56	 1.2343667e+00	 1.4694858e-01	 1.2854135e+00	 1.6868719e-01	[ 1.1472944e+00]	 2.1031642e-01


.. parsed-literal::

      57	 1.2458393e+00	 1.4610043e-01	 1.2971987e+00	 1.6823045e-01	[ 1.1492872e+00]	 2.1291280e-01


.. parsed-literal::

      58	 1.2577077e+00	 1.4449608e-01	 1.3095529e+00	 1.6779531e-01	  1.1419040e+00 	 2.0955610e-01
      59	 1.2663307e+00	 1.4371928e-01	 1.3182274e+00	 1.6775010e-01	  1.1413494e+00 	 1.9484425e-01


.. parsed-literal::

      60	 1.2747338e+00	 1.4326799e-01	 1.3262840e+00	 1.6746841e-01	[ 1.1534183e+00]	 2.1431875e-01


.. parsed-literal::

      61	 1.2879819e+00	 1.4180579e-01	 1.3393320e+00	 1.6620958e-01	[ 1.1753629e+00]	 2.0593834e-01


.. parsed-literal::

      62	 1.2961615e+00	 1.4135720e-01	 1.3474754e+00	 1.6644753e-01	[ 1.1757466e+00]	 2.0372009e-01


.. parsed-literal::

      63	 1.3042119e+00	 1.4085304e-01	 1.3557689e+00	 1.6658063e-01	[ 1.1776973e+00]	 2.1050954e-01


.. parsed-literal::

      64	 1.3178206e+00	 1.3943273e-01	 1.3705871e+00	 1.6634957e-01	  1.1650117e+00 	 2.1723270e-01


.. parsed-literal::

      65	 1.3258555e+00	 1.3933452e-01	 1.3784114e+00	 1.6686093e-01	  1.1585287e+00 	 2.2272587e-01


.. parsed-literal::

      66	 1.3399335e+00	 1.3866701e-01	 1.3927610e+00	 1.6745972e-01	  1.1190828e+00 	 2.0533061e-01
      67	 1.3474649e+00	 1.3802423e-01	 1.4005286e+00	 1.6703603e-01	  1.0932516e+00 	 1.9811058e-01


.. parsed-literal::

      68	 1.3546425e+00	 1.3762360e-01	 1.4076896e+00	 1.6674253e-01	  1.0940210e+00 	 2.0490170e-01


.. parsed-literal::

      69	 1.3618545e+00	 1.3659743e-01	 1.4151253e+00	 1.6573628e-01	  1.0932227e+00 	 2.0632625e-01


.. parsed-literal::

      70	 1.3683131e+00	 1.3585013e-01	 1.4217204e+00	 1.6522591e-01	  1.0829437e+00 	 2.2129512e-01


.. parsed-literal::

      71	 1.3748245e+00	 1.3399077e-01	 1.4282801e+00	 1.6361774e-01	  1.0924123e+00 	 2.1852326e-01


.. parsed-literal::

      72	 1.3795833e+00	 1.3351372e-01	 1.4328554e+00	 1.6331800e-01	  1.1029929e+00 	 2.2041225e-01


.. parsed-literal::

      73	 1.3850591e+00	 1.3278984e-01	 1.4383992e+00	 1.6293405e-01	  1.1031429e+00 	 2.2003007e-01
      74	 1.3915478e+00	 1.3181689e-01	 1.4449812e+00	 1.6230610e-01	  1.1091726e+00 	 1.9647908e-01


.. parsed-literal::

      75	 1.3984496e+00	 1.3062721e-01	 1.4522770e+00	 1.6131409e-01	  1.1019771e+00 	 2.1067214e-01


.. parsed-literal::

      76	 1.4049619e+00	 1.3027812e-01	 1.4588273e+00	 1.6123674e-01	  1.1006436e+00 	 2.0205903e-01


.. parsed-literal::

      77	 1.4111620e+00	 1.2978386e-01	 1.4650790e+00	 1.6091029e-01	  1.0849564e+00 	 2.1273732e-01


.. parsed-literal::

      78	 1.4171524e+00	 1.2912957e-01	 1.4711277e+00	 1.6087144e-01	  1.0438660e+00 	 2.1315527e-01


.. parsed-literal::

      79	 1.4225976e+00	 1.2828682e-01	 1.4765658e+00	 1.5966060e-01	  1.0115386e+00 	 2.1852207e-01


.. parsed-literal::

      80	 1.4261714e+00	 1.2798036e-01	 1.4800253e+00	 1.5954925e-01	  1.0116296e+00 	 2.1914101e-01
      81	 1.4312569e+00	 1.2724941e-01	 1.4851375e+00	 1.5888465e-01	  1.0044849e+00 	 1.9455242e-01


.. parsed-literal::

      82	 1.4363289e+00	 1.2644097e-01	 1.4904342e+00	 1.5829847e-01	  9.8036520e-01 	 2.0662451e-01


.. parsed-literal::

      83	 1.4415680e+00	 1.2592327e-01	 1.4956778e+00	 1.5778319e-01	  9.8199588e-01 	 2.1004677e-01


.. parsed-literal::

      84	 1.4462031e+00	 1.2532424e-01	 1.5004961e+00	 1.5732219e-01	  9.7923506e-01 	 2.0751667e-01
      85	 1.4498109e+00	 1.2506736e-01	 1.5042253e+00	 1.5735761e-01	  9.5865304e-01 	 2.1196628e-01


.. parsed-literal::

      86	 1.4539407e+00	 1.2483281e-01	 1.5084239e+00	 1.5731416e-01	  9.5338151e-01 	 2.0974636e-01


.. parsed-literal::

      87	 1.4577307e+00	 1.2465206e-01	 1.5122934e+00	 1.5750112e-01	  9.3364611e-01 	 2.0207715e-01
      88	 1.4614158e+00	 1.2459182e-01	 1.5159898e+00	 1.5768341e-01	  9.1982548e-01 	 1.9897008e-01


.. parsed-literal::

      89	 1.4646734e+00	 1.2461123e-01	 1.5192442e+00	 1.5781796e-01	  9.1975687e-01 	 2.2957349e-01


.. parsed-literal::

      90	 1.4683165e+00	 1.2453858e-01	 1.5229813e+00	 1.5774856e-01	  8.9720647e-01 	 2.1020484e-01
      91	 1.4710108e+00	 1.2448889e-01	 1.5258158e+00	 1.5763550e-01	  9.0428325e-01 	 1.9463873e-01


.. parsed-literal::

      92	 1.4738611e+00	 1.2426875e-01	 1.5286310e+00	 1.5719833e-01	  8.8098934e-01 	 2.0909882e-01


.. parsed-literal::

      93	 1.4765271e+00	 1.2402134e-01	 1.5313051e+00	 1.5665181e-01	  8.4645703e-01 	 2.0731330e-01


.. parsed-literal::

      94	 1.4787973e+00	 1.2388033e-01	 1.5335674e+00	 1.5638626e-01	  8.2347550e-01 	 2.2808552e-01
      95	 1.4839518e+00	 1.2375779e-01	 1.5387928e+00	 1.5575352e-01	  7.4613326e-01 	 2.0149422e-01


.. parsed-literal::

      96	 1.4851826e+00	 1.2355954e-01	 1.5400664e+00	 1.5579861e-01	  7.0753946e-01 	 2.0509195e-01
      97	 1.4885023e+00	 1.2357096e-01	 1.5432166e+00	 1.5593110e-01	  7.4273640e-01 	 1.9907427e-01


.. parsed-literal::

      98	 1.4897678e+00	 1.2358524e-01	 1.5444900e+00	 1.5603933e-01	  7.4449341e-01 	 2.0574450e-01


.. parsed-literal::

      99	 1.4926780e+00	 1.2358685e-01	 1.5475041e+00	 1.5638499e-01	  7.3691556e-01 	 2.1595359e-01
     100	 1.4948677e+00	 1.2344048e-01	 1.5497718e+00	 1.5658313e-01	  7.3359651e-01 	 1.8043423e-01


.. parsed-literal::

     101	 1.4974324e+00	 1.2334752e-01	 1.5523856e+00	 1.5658041e-01	  7.1609368e-01 	 1.7845702e-01
     102	 1.4996042e+00	 1.2309380e-01	 1.5545658e+00	 1.5635919e-01	  7.1313126e-01 	 1.8445659e-01


.. parsed-literal::

     103	 1.5019056e+00	 1.2269891e-01	 1.5569053e+00	 1.5583553e-01	  7.0195928e-01 	 2.1202707e-01


.. parsed-literal::

     104	 1.5035051e+00	 1.2210411e-01	 1.5586721e+00	 1.5518661e-01	  6.7071610e-01 	 2.2136736e-01
     105	 1.5067071e+00	 1.2189398e-01	 1.5617955e+00	 1.5494195e-01	  6.7674649e-01 	 1.7731810e-01


.. parsed-literal::

     106	 1.5083156e+00	 1.2176043e-01	 1.5634110e+00	 1.5484853e-01	  6.6621691e-01 	 1.9631767e-01


.. parsed-literal::

     107	 1.5102029e+00	 1.2155417e-01	 1.5653140e+00	 1.5477046e-01	  6.4282192e-01 	 2.0123458e-01


.. parsed-literal::

     108	 1.5132194e+00	 1.2119706e-01	 1.5683748e+00	 1.5476525e-01	  6.1534841e-01 	 2.1151233e-01


.. parsed-literal::

     109	 1.5147736e+00	 1.2090472e-01	 1.5699751e+00	 1.5459673e-01	  5.8485854e-01 	 3.3610320e-01


.. parsed-literal::

     110	 1.5174469e+00	 1.2060920e-01	 1.5726822e+00	 1.5463431e-01	  5.6860331e-01 	 2.1068478e-01


.. parsed-literal::

     111	 1.5192428e+00	 1.2041665e-01	 1.5744918e+00	 1.5460222e-01	  5.7810104e-01 	 2.1280217e-01
     112	 1.5212507e+00	 1.2025163e-01	 1.5765597e+00	 1.5449050e-01	  5.3934238e-01 	 1.9986725e-01


.. parsed-literal::

     113	 1.5230424e+00	 1.2006974e-01	 1.5783524e+00	 1.5434651e-01	  5.2532658e-01 	 2.0866966e-01


.. parsed-literal::

     114	 1.5251188e+00	 1.1981717e-01	 1.5804610e+00	 1.5418379e-01	  4.5484811e-01 	 2.1188760e-01


.. parsed-literal::

     115	 1.5268558e+00	 1.1957711e-01	 1.5822395e+00	 1.5402330e-01	  3.6658200e-01 	 2.1131277e-01


.. parsed-literal::

     116	 1.5285348e+00	 1.1933965e-01	 1.5839585e+00	 1.5394897e-01	  3.2110915e-01 	 2.1225095e-01
     117	 1.5303116e+00	 1.1909939e-01	 1.5857846e+00	 1.5397226e-01	  2.8672197e-01 	 1.9101453e-01


.. parsed-literal::

     118	 1.5316561e+00	 1.1887207e-01	 1.5872195e+00	 1.5393892e-01	  2.3932611e-01 	 2.0980072e-01


.. parsed-literal::

     119	 1.5332536e+00	 1.1879616e-01	 1.5888016e+00	 1.5392825e-01	  2.5555029e-01 	 2.1062517e-01


.. parsed-literal::

     120	 1.5351474e+00	 1.1871542e-01	 1.5907094e+00	 1.5387080e-01	  2.6117886e-01 	 2.0803118e-01


.. parsed-literal::

     121	 1.5366709e+00	 1.1864188e-01	 1.5922343e+00	 1.5380279e-01	  2.3645404e-01 	 2.1285844e-01


.. parsed-literal::

     122	 1.5393916e+00	 1.1856440e-01	 1.5949460e+00	 1.5394294e-01	  2.0439603e-01 	 2.1328831e-01


.. parsed-literal::

     123	 1.5407017e+00	 1.1847064e-01	 1.5962433e+00	 1.5390272e-01	  1.6449737e-01 	 3.3582020e-01


.. parsed-literal::

     124	 1.5421921e+00	 1.1845786e-01	 1.5977147e+00	 1.5402173e-01	  1.4516822e-01 	 2.1286225e-01


.. parsed-literal::

     125	 1.5436791e+00	 1.1842574e-01	 1.5992068e+00	 1.5413128e-01	  1.3438653e-01 	 2.1153736e-01


.. parsed-literal::

     126	 1.5452228e+00	 1.1841228e-01	 1.6008235e+00	 1.5426140e-01	  1.0007736e-01 	 2.1356583e-01
     127	 1.5466477e+00	 1.1835823e-01	 1.6023064e+00	 1.5430734e-01	  1.0105738e-01 	 2.0338798e-01


.. parsed-literal::

     128	 1.5477887e+00	 1.1834809e-01	 1.6034433e+00	 1.5429025e-01	  9.2677514e-02 	 2.1227098e-01
     129	 1.5492974e+00	 1.1837645e-01	 1.6049916e+00	 1.5431703e-01	  4.1519033e-02 	 1.9093990e-01


.. parsed-literal::

     130	 1.5504278e+00	 1.1835160e-01	 1.6061888e+00	 1.5433400e-01	 -5.4676499e-03 	 2.1580648e-01
     131	 1.5518349e+00	 1.1838359e-01	 1.6076185e+00	 1.5448499e-01	 -5.3144407e-02 	 1.8847895e-01


.. parsed-literal::

     132	 1.5539916e+00	 1.1838663e-01	 1.6098409e+00	 1.5475977e-01	 -1.5657594e-01 	 1.9270658e-01
     133	 1.5552893e+00	 1.1833942e-01	 1.6112114e+00	 1.5495163e-01	 -1.8863202e-01 	 1.8791533e-01


.. parsed-literal::

     134	 1.5568376e+00	 1.1822041e-01	 1.6128190e+00	 1.5511987e-01	 -2.6041765e-01 	 2.0777559e-01


.. parsed-literal::

     135	 1.5582929e+00	 1.1804506e-01	 1.6143470e+00	 1.5516618e-01	 -2.8908856e-01 	 2.2090673e-01


.. parsed-literal::

     136	 1.5597209e+00	 1.1791858e-01	 1.6158550e+00	 1.5532540e-01	 -3.9136886e-01 	 2.0903707e-01
     137	 1.5609956e+00	 1.1790583e-01	 1.6171065e+00	 1.5538960e-01	 -4.1105546e-01 	 1.9451213e-01


.. parsed-literal::

     138	 1.5627954e+00	 1.1795647e-01	 1.6189328e+00	 1.5565936e-01	 -4.8928530e-01 	 2.1115851e-01
     139	 1.5629696e+00	 1.1801047e-01	 1.6191835e+00	 1.5595658e-01	 -5.2409830e-01 	 1.8656373e-01


.. parsed-literal::

     140	 1.5643456e+00	 1.1795012e-01	 1.6204896e+00	 1.5588075e-01	 -5.2956485e-01 	 2.0345664e-01


.. parsed-literal::

     141	 1.5651366e+00	 1.1787825e-01	 1.6212656e+00	 1.5587455e-01	 -5.4097337e-01 	 2.1037912e-01
     142	 1.5661485e+00	 1.1773844e-01	 1.6222824e+00	 1.5585440e-01	 -5.6520115e-01 	 1.9521832e-01


.. parsed-literal::

     143	 1.5671934e+00	 1.1758038e-01	 1.6233949e+00	 1.5590945e-01	 -6.0910602e-01 	 2.0020604e-01
     144	 1.5682995e+00	 1.1724660e-01	 1.6245854e+00	 1.5569439e-01	 -6.7648970e-01 	 1.7552900e-01


.. parsed-literal::

     145	 1.5691474e+00	 1.1717461e-01	 1.6254390e+00	 1.5563527e-01	 -6.8270203e-01 	 1.9707179e-01
     146	 1.5702348e+00	 1.1702572e-01	 1.6265813e+00	 1.5556220e-01	 -7.4263521e-01 	 1.7924595e-01


.. parsed-literal::

     147	 1.5712674e+00	 1.1684695e-01	 1.6276596e+00	 1.5550810e-01	 -7.8128875e-01 	 2.0098662e-01
     148	 1.5726417e+00	 1.1662197e-01	 1.6290968e+00	 1.5558466e-01	 -9.2714270e-01 	 1.8885112e-01


.. parsed-literal::

     149	 1.5736808e+00	 1.1647688e-01	 1.6301251e+00	 1.5556037e-01	 -9.7451544e-01 	 2.0981622e-01
     150	 1.5753077e+00	 1.1622948e-01	 1.6317257e+00	 1.5552746e-01	 -1.0875618e+00 	 1.7272997e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.1 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fdac863e1d0>



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
    CPU times: user 1.84 s, sys: 54.9 ms, total: 1.9 s
    Wall time: 644 ms


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

