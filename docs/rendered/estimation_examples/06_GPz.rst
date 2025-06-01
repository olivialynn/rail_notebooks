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
       1	-3.4281176e-01	 3.2063753e-01	-3.3304311e-01	 3.2194621e-01	[-3.3418920e-01]	 4.6381545e-01


.. parsed-literal::

       2	-2.7297870e-01	 3.1007416e-01	-2.4915892e-01	 3.1009058e-01	[-2.4970564e-01]	 2.3007178e-01


.. parsed-literal::

       3	-2.2931163e-01	 2.8990162e-01	-1.8770269e-01	 2.9178959e-01	[-1.9179747e-01]	 2.7682495e-01
       4	-1.9945611e-01	 2.6710201e-01	-1.5905430e-01	 2.6361968e-01	[-1.3742454e-01]	 1.9823956e-01


.. parsed-literal::

       5	-1.1121919e-01	 2.5874224e-01	-7.6561500e-02	 2.5709663e-01	[-6.7824109e-02]	 1.9257689e-01


.. parsed-literal::

       6	-7.3287065e-02	 2.5209872e-01	-4.2260031e-02	 2.4962513e-01	[-3.3300368e-02]	 2.0189381e-01
       7	-5.5321275e-02	 2.4954035e-01	-3.1110363e-02	 2.4673355e-01	[-2.0758642e-02]	 2.0173430e-01


.. parsed-literal::

       8	-4.0678595e-02	 2.4696985e-01	-2.0427516e-02	 2.4431386e-01	[-1.0656593e-02]	 1.9914770e-01


.. parsed-literal::

       9	-2.6013843e-02	 2.4424062e-01	-8.8934140e-03	 2.4195912e-01	[-4.1168023e-04]	 2.1454430e-01


.. parsed-literal::

      10	-1.8729515e-02	 2.4307365e-01	-3.8808912e-03	 2.4218310e-01	 -7.0235536e-04 	 2.1387911e-01


.. parsed-literal::

      11	-1.1787423e-02	 2.4171585e-01	 2.5190185e-03	 2.4057881e-01	[ 6.3039792e-03]	 2.0950055e-01


.. parsed-literal::

      12	-9.0674554e-03	 2.4121088e-01	 4.9715596e-03	 2.3983631e-01	[ 9.7371849e-03]	 2.1173382e-01


.. parsed-literal::

      13	-5.2724320e-03	 2.4045431e-01	 8.8448920e-03	 2.3886845e-01	[ 1.4668111e-02]	 2.1096897e-01


.. parsed-literal::

      14	 5.1346354e-02	 2.2862850e-01	 6.7400852e-02	 2.2608515e-01	[ 7.5250207e-02]	 3.2472014e-01


.. parsed-literal::

      15	 8.9814864e-02	 2.2360353e-01	 1.0881731e-01	 2.2062491e-01	[ 1.1622005e-01]	 2.0692682e-01


.. parsed-literal::

      16	 1.9061705e-01	 2.2080639e-01	 2.1234080e-01	 2.1918982e-01	[ 2.1847928e-01]	 2.0250869e-01


.. parsed-literal::

      17	 2.7652544e-01	 2.1715822e-01	 3.0471275e-01	 2.1513833e-01	[ 3.0317433e-01]	 2.1306205e-01


.. parsed-literal::

      18	 3.1704432e-01	 2.1476354e-01	 3.4748486e-01	 2.1515294e-01	[ 3.3857982e-01]	 2.1825647e-01
      19	 3.9064088e-01	 2.1267097e-01	 4.2167569e-01	 2.1274313e-01	[ 4.0923217e-01]	 1.7816973e-01


.. parsed-literal::

      20	 4.9370553e-01	 2.1338704e-01	 5.2659630e-01	 2.1536171e-01	[ 5.0756056e-01]	 2.0535874e-01


.. parsed-literal::

      21	 5.5782009e-01	 2.1491793e-01	 5.9361284e-01	 2.1495443e-01	[ 5.6787617e-01]	 2.0428801e-01


.. parsed-literal::

      22	 5.9034697e-01	 2.1190222e-01	 6.2927770e-01	 2.0998870e-01	[ 6.0205559e-01]	 2.0894742e-01


.. parsed-literal::

      23	 6.2469483e-01	 2.0754633e-01	 6.6271371e-01	 2.0750910e-01	[ 6.2869507e-01]	 2.1519852e-01


.. parsed-literal::

      24	 6.5631158e-01	 2.0480736e-01	 6.9286455e-01	 2.0656173e-01	[ 6.5797645e-01]	 2.1629834e-01
      25	 6.9694014e-01	 2.0096391e-01	 7.3195070e-01	 2.0329124e-01	[ 7.0648294e-01]	 1.9170880e-01


.. parsed-literal::

      26	 7.3975953e-01	 1.9659404e-01	 7.7532496e-01	 1.9960308e-01	[ 7.4903354e-01]	 1.9887590e-01


.. parsed-literal::

      27	 7.6820075e-01	 1.9730531e-01	 8.0562276e-01	 2.0295372e-01	[ 7.7812957e-01]	 2.0579433e-01
      28	 7.8646682e-01	 1.9644355e-01	 8.2476556e-01	 2.0311436e-01	[ 7.9896977e-01]	 2.0260429e-01


.. parsed-literal::

      29	 7.9961131e-01	 1.9593000e-01	 8.3767758e-01	 2.0296490e-01	[ 8.1194689e-01]	 2.0270586e-01


.. parsed-literal::

      30	 8.2650281e-01	 1.9416032e-01	 8.6618621e-01	 1.9852714e-01	[ 8.4470054e-01]	 2.1506763e-01


.. parsed-literal::

      31	 8.6179522e-01	 1.9313292e-01	 9.0221299e-01	 1.9738157e-01	[ 8.7232828e-01]	 2.0656919e-01


.. parsed-literal::

      32	 8.8672445e-01	 1.9126830e-01	 9.2872749e-01	 1.9432057e-01	[ 9.0528828e-01]	 2.1171212e-01
      33	 9.1211053e-01	 1.8631313e-01	 9.5357568e-01	 1.8785202e-01	[ 9.2600117e-01]	 1.8396544e-01


.. parsed-literal::

      34	 9.2462513e-01	 1.8379210e-01	 9.6607071e-01	 1.8442203e-01	[ 9.3638247e-01]	 2.0319939e-01


.. parsed-literal::

      35	 9.3569990e-01	 1.8348733e-01	 9.7683725e-01	 1.8243948e-01	[ 9.5173909e-01]	 2.0285153e-01


.. parsed-literal::

      36	 9.4681553e-01	 1.8303049e-01	 9.8861763e-01	 1.8284855e-01	[ 9.6243644e-01]	 2.0654726e-01


.. parsed-literal::

      37	 9.5500168e-01	 1.8296666e-01	 9.9748093e-01	 1.8300379e-01	[ 9.6803519e-01]	 2.0780492e-01


.. parsed-literal::

      38	 9.7008736e-01	 1.8139161e-01	 1.0140311e+00	 1.8310624e-01	[ 9.7376005e-01]	 2.0498419e-01


.. parsed-literal::

      39	 9.8157404e-01	 1.8130963e-01	 1.0259191e+00	 1.8392232e-01	[ 9.8496675e-01]	 2.0935965e-01


.. parsed-literal::

      40	 9.9470284e-01	 1.7915961e-01	 1.0388487e+00	 1.8329849e-01	[ 9.9545051e-01]	 2.0500803e-01


.. parsed-literal::

      41	 1.0070365e+00	 1.7746888e-01	 1.0511841e+00	 1.8270014e-01	[ 1.0052105e+00]	 2.1915102e-01


.. parsed-literal::

      42	 1.0174230e+00	 1.7673650e-01	 1.0617698e+00	 1.8172630e-01	[ 1.0126563e+00]	 2.0619535e-01


.. parsed-literal::

      43	 1.0272226e+00	 1.7633330e-01	 1.0722208e+00	 1.8105328e-01	[ 1.0218403e+00]	 2.1147513e-01


.. parsed-literal::

      44	 1.0361851e+00	 1.7553710e-01	 1.0813847e+00	 1.8024951e-01	[ 1.0273123e+00]	 2.0391893e-01


.. parsed-literal::

      45	 1.0417903e+00	 1.7532258e-01	 1.0870606e+00	 1.8005763e-01	[ 1.0326486e+00]	 2.0390749e-01


.. parsed-literal::

      46	 1.0539895e+00	 1.7415472e-01	 1.0997872e+00	 1.7880817e-01	[ 1.0410523e+00]	 2.0982099e-01
      47	 1.0614699e+00	 1.7343183e-01	 1.1078642e+00	 1.7819030e-01	[ 1.0488463e+00]	 1.8295789e-01


.. parsed-literal::

      48	 1.0715142e+00	 1.7248133e-01	 1.1178072e+00	 1.7752322e-01	[ 1.0630720e+00]	 2.0920181e-01


.. parsed-literal::

      49	 1.0766935e+00	 1.7135614e-01	 1.1227410e+00	 1.7682050e-01	[ 1.0678639e+00]	 2.0581174e-01


.. parsed-literal::

      50	 1.0867269e+00	 1.6940642e-01	 1.1330093e+00	 1.7529129e-01	[ 1.0743059e+00]	 2.1757364e-01


.. parsed-literal::

      51	 1.0912897e+00	 1.6729551e-01	 1.1379961e+00	 1.7310528e-01	[ 1.0774316e+00]	 2.1199179e-01


.. parsed-literal::

      52	 1.0993574e+00	 1.6671168e-01	 1.1460203e+00	 1.7236008e-01	[ 1.0861136e+00]	 2.1711898e-01


.. parsed-literal::

      53	 1.1044457e+00	 1.6616661e-01	 1.1511870e+00	 1.7140235e-01	[ 1.0930460e+00]	 2.1230459e-01


.. parsed-literal::

      54	 1.1125762e+00	 1.6487266e-01	 1.1594663e+00	 1.6931942e-01	[ 1.1040500e+00]	 2.0700979e-01


.. parsed-literal::

      55	 1.1265242e+00	 1.6272140e-01	 1.1739206e+00	 1.6614280e-01	[ 1.1201586e+00]	 2.1792626e-01


.. parsed-literal::

      56	 1.1343868e+00	 1.6028891e-01	 1.1821653e+00	 1.6228758e-01	[ 1.1289406e+00]	 3.2529092e-01
      57	 1.1415140e+00	 1.5951055e-01	 1.1897211e+00	 1.6074091e-01	[ 1.1334955e+00]	 1.8626785e-01


.. parsed-literal::

      58	 1.1477979e+00	 1.5936767e-01	 1.1960900e+00	 1.6069655e-01	  1.1332543e+00 	 2.0571947e-01
      59	 1.1557447e+00	 1.5851190e-01	 1.2044930e+00	 1.5869804e-01	[ 1.1345520e+00]	 1.8713379e-01


.. parsed-literal::

      60	 1.1629205e+00	 1.5846415e-01	 1.2119251e+00	 1.5856799e-01	  1.1317358e+00 	 2.0897412e-01


.. parsed-literal::

      61	 1.1697793e+00	 1.5812746e-01	 1.2189748e+00	 1.5810884e-01	[ 1.1353428e+00]	 2.1794033e-01
      62	 1.1787261e+00	 1.5739626e-01	 1.2283751e+00	 1.5734455e-01	[ 1.1363074e+00]	 2.0059133e-01


.. parsed-literal::

      63	 1.1845981e+00	 1.5609129e-01	 1.2346508e+00	 1.5747197e-01	[ 1.1407038e+00]	 1.9033360e-01
      64	 1.1913605e+00	 1.5537112e-01	 1.2412923e+00	 1.5643313e-01	[ 1.1479425e+00]	 1.9253969e-01


.. parsed-literal::

      65	 1.1980681e+00	 1.5442269e-01	 1.2481672e+00	 1.5535324e-01	[ 1.1495946e+00]	 2.1991920e-01


.. parsed-literal::

      66	 1.2047833e+00	 1.5355423e-01	 1.2550757e+00	 1.5446070e-01	[ 1.1529053e+00]	 2.1032715e-01
      67	 1.2134209e+00	 1.5150724e-01	 1.2642973e+00	 1.5190749e-01	[ 1.1640845e+00]	 1.8273807e-01


.. parsed-literal::

      68	 1.2216049e+00	 1.5089858e-01	 1.2724549e+00	 1.5152171e-01	[ 1.1721765e+00]	 2.0510507e-01


.. parsed-literal::

      69	 1.2254268e+00	 1.5097249e-01	 1.2761125e+00	 1.5153567e-01	[ 1.1750862e+00]	 2.2017550e-01


.. parsed-literal::

      70	 1.2309460e+00	 1.5032521e-01	 1.2817613e+00	 1.5065513e-01	[ 1.1815615e+00]	 2.1915460e-01


.. parsed-literal::

      71	 1.2346345e+00	 1.5044143e-01	 1.2855443e+00	 1.5051265e-01	  1.1782866e+00 	 2.0494294e-01


.. parsed-literal::

      72	 1.2393724e+00	 1.4984487e-01	 1.2901646e+00	 1.4964731e-01	[ 1.1861565e+00]	 2.1027422e-01


.. parsed-literal::

      73	 1.2437758e+00	 1.4919536e-01	 1.2944853e+00	 1.4863605e-01	[ 1.1934107e+00]	 2.0708942e-01


.. parsed-literal::

      74	 1.2475254e+00	 1.4883335e-01	 1.2981323e+00	 1.4800133e-01	[ 1.1976855e+00]	 2.1134734e-01


.. parsed-literal::

      75	 1.2554039e+00	 1.4791569e-01	 1.3061197e+00	 1.4663866e-01	[ 1.2045596e+00]	 2.0820451e-01


.. parsed-literal::

      76	 1.2597962e+00	 1.4730170e-01	 1.3104590e+00	 1.4572769e-01	[ 1.2088225e+00]	 3.3049893e-01


.. parsed-literal::

      77	 1.2642342e+00	 1.4711054e-01	 1.3148903e+00	 1.4545170e-01	[ 1.2104378e+00]	 2.1661997e-01


.. parsed-literal::

      78	 1.2696012e+00	 1.4693751e-01	 1.3203770e+00	 1.4506477e-01	[ 1.2126904e+00]	 2.1374917e-01
      79	 1.2755361e+00	 1.4699188e-01	 1.3264606e+00	 1.4462427e-01	[ 1.2156180e+00]	 1.8253684e-01


.. parsed-literal::

      80	 1.2783054e+00	 1.4651795e-01	 1.3299099e+00	 1.4371253e-01	[ 1.2172107e+00]	 2.0322561e-01
      81	 1.2879277e+00	 1.4678705e-01	 1.3392158e+00	 1.4329895e-01	[ 1.2277324e+00]	 1.8409467e-01


.. parsed-literal::

      82	 1.2911418e+00	 1.4649994e-01	 1.3424065e+00	 1.4296873e-01	[ 1.2329495e+00]	 2.1472549e-01
      83	 1.2977942e+00	 1.4603082e-01	 1.3492523e+00	 1.4252592e-01	[ 1.2428656e+00]	 1.8024874e-01


.. parsed-literal::

      84	 1.3012433e+00	 1.4572700e-01	 1.3532127e+00	 1.4297498e-01	[ 1.2533306e+00]	 2.1567369e-01


.. parsed-literal::

      85	 1.3075756e+00	 1.4556445e-01	 1.3594211e+00	 1.4271245e-01	[ 1.2550561e+00]	 2.1192741e-01


.. parsed-literal::

      86	 1.3110785e+00	 1.4556139e-01	 1.3629973e+00	 1.4267865e-01	  1.2533429e+00 	 2.1645522e-01


.. parsed-literal::

      87	 1.3144699e+00	 1.4563858e-01	 1.3665754e+00	 1.4275597e-01	  1.2517956e+00 	 2.1039224e-01


.. parsed-literal::

      88	 1.3198821e+00	 1.4568937e-01	 1.3723274e+00	 1.4242434e-01	  1.2489279e+00 	 2.1067882e-01


.. parsed-literal::

      89	 1.3214006e+00	 1.4615137e-01	 1.3744143e+00	 1.4224873e-01	  1.2407420e+00 	 2.1153784e-01


.. parsed-literal::

      90	 1.3269031e+00	 1.4570021e-01	 1.3795518e+00	 1.4193234e-01	  1.2517029e+00 	 2.0184731e-01
      91	 1.3291768e+00	 1.4550822e-01	 1.3817346e+00	 1.4156766e-01	[ 1.2555230e+00]	 1.7709923e-01


.. parsed-literal::

      92	 1.3327067e+00	 1.4526206e-01	 1.3853138e+00	 1.4101715e-01	[ 1.2588916e+00]	 1.9540930e-01
      93	 1.3372885e+00	 1.4497293e-01	 1.3900454e+00	 1.4031129e-01	  1.2584963e+00 	 1.8980813e-01


.. parsed-literal::

      94	 1.3415209e+00	 1.4481248e-01	 1.3945648e+00	 1.3996916e-01	  1.2574031e+00 	 2.1018934e-01
      95	 1.3453566e+00	 1.4466774e-01	 1.3983456e+00	 1.4011183e-01	[ 1.2591802e+00]	 2.1059585e-01


.. parsed-literal::

      96	 1.3504731e+00	 1.4452552e-01	 1.4034863e+00	 1.4047347e-01	[ 1.2607375e+00]	 1.9782925e-01


.. parsed-literal::

      97	 1.3535401e+00	 1.4455322e-01	 1.4066571e+00	 1.4076284e-01	[ 1.2626116e+00]	 2.0588088e-01


.. parsed-literal::

      98	 1.3571940e+00	 1.4443761e-01	 1.4102520e+00	 1.4060699e-01	[ 1.2689900e+00]	 2.1428800e-01


.. parsed-literal::

      99	 1.3613412e+00	 1.4423946e-01	 1.4143467e+00	 1.4000003e-01	[ 1.2787062e+00]	 2.2116828e-01


.. parsed-literal::

     100	 1.3650894e+00	 1.4416961e-01	 1.4180256e+00	 1.3942556e-01	[ 1.2875193e+00]	 2.1181536e-01


.. parsed-literal::

     101	 1.3687149e+00	 1.4422874e-01	 1.4216784e+00	 1.3900062e-01	[ 1.2936036e+00]	 2.0561123e-01


.. parsed-literal::

     102	 1.3725017e+00	 1.4416369e-01	 1.4255521e+00	 1.3876172e-01	[ 1.2964266e+00]	 2.0439696e-01


.. parsed-literal::

     103	 1.3753688e+00	 1.4407670e-01	 1.4286480e+00	 1.3856405e-01	[ 1.2968106e+00]	 2.1114945e-01


.. parsed-literal::

     104	 1.3786798e+00	 1.4368105e-01	 1.4319369e+00	 1.3867884e-01	[ 1.2981091e+00]	 2.0390058e-01
     105	 1.3817969e+00	 1.4330080e-01	 1.4350673e+00	 1.3867979e-01	[ 1.2995759e+00]	 1.8800211e-01


.. parsed-literal::

     106	 1.3846144e+00	 1.4312290e-01	 1.4379771e+00	 1.3885093e-01	[ 1.3047025e+00]	 2.0636725e-01
     107	 1.3877254e+00	 1.4287219e-01	 1.4410359e+00	 1.3855495e-01	  1.3038418e+00 	 1.6271472e-01


.. parsed-literal::

     108	 1.3900450e+00	 1.4277479e-01	 1.4433423e+00	 1.3836056e-01	[ 1.3062419e+00]	 1.8591237e-01
     109	 1.3947120e+00	 1.4254018e-01	 1.4481315e+00	 1.3805258e-01	[ 1.3093430e+00]	 1.8540549e-01


.. parsed-literal::

     110	 1.3974629e+00	 1.4224788e-01	 1.4510062e+00	 1.3823796e-01	  1.3092112e+00 	 2.1330166e-01


.. parsed-literal::

     111	 1.4012103e+00	 1.4194109e-01	 1.4547924e+00	 1.3789634e-01	[ 1.3116939e+00]	 2.2110772e-01


.. parsed-literal::

     112	 1.4049310e+00	 1.4142552e-01	 1.4585385e+00	 1.3764559e-01	[ 1.3121152e+00]	 2.1998549e-01


.. parsed-literal::

     113	 1.4074883e+00	 1.4114534e-01	 1.4611543e+00	 1.3751777e-01	[ 1.3124560e+00]	 2.0259428e-01
     114	 1.4108131e+00	 1.4076345e-01	 1.4644910e+00	 1.3744450e-01	[ 1.3130728e+00]	 1.9379354e-01


.. parsed-literal::

     115	 1.4146646e+00	 1.4055331e-01	 1.4684111e+00	 1.3746494e-01	[ 1.3131285e+00]	 1.7915797e-01


.. parsed-literal::

     116	 1.4174876e+00	 1.4002849e-01	 1.4713239e+00	 1.3711514e-01	[ 1.3146461e+00]	 2.2882915e-01


.. parsed-literal::

     117	 1.4197355e+00	 1.4009587e-01	 1.4735394e+00	 1.3711693e-01	[ 1.3155302e+00]	 2.1102190e-01


.. parsed-literal::

     118	 1.4226897e+00	 1.4003536e-01	 1.4765593e+00	 1.3697165e-01	  1.3137713e+00 	 2.2198820e-01


.. parsed-literal::

     119	 1.4248071e+00	 1.3992878e-01	 1.4787878e+00	 1.3716873e-01	  1.3108044e+00 	 2.1375847e-01


.. parsed-literal::

     120	 1.4273079e+00	 1.3966142e-01	 1.4813505e+00	 1.3705189e-01	  1.3074352e+00 	 2.1385193e-01
     121	 1.4298881e+00	 1.3936498e-01	 1.4839378e+00	 1.3700854e-01	  1.3042326e+00 	 1.9037557e-01


.. parsed-literal::

     122	 1.4320511e+00	 1.3930652e-01	 1.4861410e+00	 1.3720239e-01	  1.3059960e+00 	 1.7664599e-01


.. parsed-literal::

     123	 1.4344514e+00	 1.3910013e-01	 1.4885752e+00	 1.3724188e-01	  1.3027578e+00 	 2.1699762e-01


.. parsed-literal::

     124	 1.4370493e+00	 1.3909231e-01	 1.4911563e+00	 1.3726254e-01	  1.3048991e+00 	 2.1551037e-01


.. parsed-literal::

     125	 1.4405825e+00	 1.3896311e-01	 1.4948701e+00	 1.3748349e-01	  1.3011438e+00 	 2.0340204e-01


.. parsed-literal::

     126	 1.4427869e+00	 1.3887732e-01	 1.4971985e+00	 1.3720433e-01	  1.2989327e+00 	 2.1573019e-01


.. parsed-literal::

     127	 1.4444020e+00	 1.3870171e-01	 1.4987549e+00	 1.3717946e-01	  1.3000581e+00 	 2.2601557e-01


.. parsed-literal::

     128	 1.4467904e+00	 1.3824993e-01	 1.5011957e+00	 1.3707159e-01	  1.2968919e+00 	 2.0905280e-01


.. parsed-literal::

     129	 1.4490220e+00	 1.3796414e-01	 1.5034328e+00	 1.3699907e-01	  1.2988566e+00 	 2.1527314e-01
     130	 1.4520790e+00	 1.3753027e-01	 1.5064304e+00	 1.3676841e-01	  1.3032369e+00 	 1.9427752e-01


.. parsed-literal::

     131	 1.4541709e+00	 1.3739818e-01	 1.5085345e+00	 1.3679964e-01	  1.3067149e+00 	 2.1910620e-01
     132	 1.4560315e+00	 1.3735199e-01	 1.5103779e+00	 1.3661785e-01	  1.3087149e+00 	 1.8702769e-01


.. parsed-literal::

     133	 1.4579377e+00	 1.3731024e-01	 1.5123194e+00	 1.3654060e-01	  1.3086388e+00 	 2.0525813e-01


.. parsed-literal::

     134	 1.4599795e+00	 1.3715874e-01	 1.5145628e+00	 1.3613729e-01	  1.3026448e+00 	 2.1177816e-01


.. parsed-literal::

     135	 1.4620221e+00	 1.3703405e-01	 1.5166223e+00	 1.3614967e-01	  1.2993690e+00 	 2.1449256e-01


.. parsed-literal::

     136	 1.4633977e+00	 1.3692929e-01	 1.5180161e+00	 1.3609971e-01	  1.2981171e+00 	 2.1804881e-01
     137	 1.4650416e+00	 1.3679850e-01	 1.5196093e+00	 1.3602045e-01	  1.2971013e+00 	 1.9750714e-01


.. parsed-literal::

     138	 1.4670330e+00	 1.3680216e-01	 1.5215295e+00	 1.3577182e-01	  1.2964264e+00 	 2.0712757e-01


.. parsed-literal::

     139	 1.4688390e+00	 1.3670586e-01	 1.5232408e+00	 1.3576322e-01	  1.2966412e+00 	 2.1974182e-01


.. parsed-literal::

     140	 1.4702364e+00	 1.3662825e-01	 1.5245765e+00	 1.3574399e-01	  1.2974997e+00 	 2.1692967e-01
     141	 1.4722179e+00	 1.3643792e-01	 1.5266024e+00	 1.3571974e-01	  1.2954428e+00 	 2.0015764e-01


.. parsed-literal::

     142	 1.4729381e+00	 1.3594026e-01	 1.5277079e+00	 1.3561083e-01	  1.2866233e+00 	 2.2104883e-01


.. parsed-literal::

     143	 1.4757175e+00	 1.3594610e-01	 1.5303528e+00	 1.3567316e-01	  1.2893483e+00 	 2.1248746e-01


.. parsed-literal::

     144	 1.4767304e+00	 1.3587122e-01	 1.5314278e+00	 1.3569169e-01	  1.2880565e+00 	 2.1400356e-01


.. parsed-literal::

     145	 1.4789575e+00	 1.3569145e-01	 1.5338219e+00	 1.3574610e-01	  1.2890674e+00 	 2.0909333e-01
     146	 1.4808004e+00	 1.3546830e-01	 1.5359438e+00	 1.3565834e-01	  1.2891486e+00 	 1.7583323e-01


.. parsed-literal::

     147	 1.4835812e+00	 1.3539797e-01	 1.5386151e+00	 1.3584957e-01	  1.3020103e+00 	 2.1507788e-01


.. parsed-literal::

     148	 1.4851286e+00	 1.3540444e-01	 1.5400449e+00	 1.3591559e-01	  1.3110441e+00 	 2.1438384e-01


.. parsed-literal::

     149	 1.4867099e+00	 1.3539093e-01	 1.5415370e+00	 1.3599767e-01	[ 1.3185355e+00]	 2.1901584e-01


.. parsed-literal::

     150	 1.4875404e+00	 1.3532390e-01	 1.5425905e+00	 1.3616498e-01	  1.3157980e+00 	 2.0223880e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.25 s, total: 2min 6s
    Wall time: 31.8 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fc7fc598f70>



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
    CPU times: user 1.78 s, sys: 39 ms, total: 1.82 s
    Wall time: 566 ms


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

