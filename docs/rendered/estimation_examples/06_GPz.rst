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
       1	-3.4166454e-01	 3.2019162e-01	-3.3199370e-01	 3.2190516e-01	[-3.3553260e-01]	 4.5954847e-01


.. parsed-literal::

       2	-2.7184093e-01	 3.0971102e-01	-2.4798775e-01	 3.1188812e-01	[-2.5487408e-01]	 2.3170304e-01


.. parsed-literal::

       3	-2.2843046e-01	 2.8937859e-01	-1.8628201e-01	 2.9108398e-01	[-1.9540194e-01]	 2.9415584e-01


.. parsed-literal::

       4	-1.8992762e-01	 2.6526175e-01	-1.4735950e-01	 2.6836449e-01	[-1.6653402e-01]	 2.0808911e-01
       5	-1.0116531e-01	 2.5590331e-01	-6.6815939e-02	 2.5877373e-01	[-8.1124242e-02]	 2.0272565e-01


.. parsed-literal::

       6	-6.9585759e-02	 2.5154565e-01	-3.9207887e-02	 2.5565534e-01	[-5.3713954e-02]	 1.7791176e-01
       7	-5.1875223e-02	 2.4859554e-01	-2.7443647e-02	 2.5169932e-01	[-3.8817182e-02]	 1.8759060e-01


.. parsed-literal::

       8	-4.0315283e-02	 2.4676419e-01	-1.9828792e-02	 2.4901310e-01	[-2.8167446e-02]	 1.8221116e-01
       9	-2.8825384e-02	 2.4474670e-01	-1.1171442e-02	 2.4610700e-01	[-1.6021192e-02]	 1.9518518e-01


.. parsed-literal::

      10	-1.8441174e-02	 2.4256524e-01	-2.9302368e-03	 2.4355979e-01	[-6.5198716e-03]	 2.1881604e-01


.. parsed-literal::

      11	-1.4084461e-02	 2.4186683e-01	-3.2358301e-04	 2.4149483e-01	[ 1.5503116e-03]	 2.1816349e-01


.. parsed-literal::

      12	-9.1294635e-03	 2.4136128e-01	 4.6022758e-03	 2.4073284e-01	[ 7.7467411e-03]	 2.1057892e-01


.. parsed-literal::

      13	-6.4335028e-03	 2.4074784e-01	 7.3955134e-03	 2.3997659e-01	[ 1.1369867e-02]	 2.2063327e-01


.. parsed-literal::

      14	-2.6789694e-03	 2.3989123e-01	 1.1713608e-02	 2.3888510e-01	[ 1.6805935e-02]	 2.0923233e-01


.. parsed-literal::

      15	 1.4298586e-01	 2.2521518e-01	 1.6666277e-01	 2.2223529e-01	[ 1.7955402e-01]	 2.9678798e-01


.. parsed-literal::

      16	 1.8111416e-01	 2.2318752e-01	 2.0489384e-01	 2.2202682e-01	[ 2.0950109e-01]	 3.2164669e-01


.. parsed-literal::

      17	 2.2882925e-01	 2.1820715e-01	 2.5444413e-01	 2.1703866e-01	[ 2.6094107e-01]	 2.1166587e-01


.. parsed-literal::

      18	 3.1965444e-01	 2.1177147e-01	 3.5027751e-01	 2.1133606e-01	[ 3.5768437e-01]	 2.0574427e-01
      19	 3.4442078e-01	 2.1030647e-01	 3.7717481e-01	 2.1080245e-01	[ 3.8132771e-01]	 2.0122266e-01


.. parsed-literal::

      20	 3.9443009e-01	 2.0854858e-01	 4.2748281e-01	 2.1126537e-01	[ 4.2765250e-01]	 2.0318675e-01
      21	 4.3233147e-01	 2.0826490e-01	 4.6580112e-01	 2.1144082e-01	[ 4.6591385e-01]	 1.8583226e-01


.. parsed-literal::

      22	 5.2324965e-01	 2.0809992e-01	 5.5812580e-01	 2.0839361e-01	[ 5.6715174e-01]	 2.1525264e-01


.. parsed-literal::

      23	 5.6552215e-01	 2.1483165e-01	 6.0441543e-01	 2.1732115e-01	[ 6.0485228e-01]	 2.1446872e-01
      24	 6.1808336e-01	 2.0624782e-01	 6.5670990e-01	 2.0666865e-01	[ 6.5398381e-01]	 1.8503499e-01


.. parsed-literal::

      25	 6.4984965e-01	 2.0190990e-01	 6.8743916e-01	 2.0153099e-01	[ 6.8249876e-01]	 1.9417548e-01


.. parsed-literal::

      26	 6.7328519e-01	 1.9974305e-01	 7.0972159e-01	 1.9815002e-01	[ 6.9679466e-01]	 2.0452142e-01


.. parsed-literal::

      27	 6.9895635e-01	 1.9965140e-01	 7.3499371e-01	 1.9805112e-01	[ 7.2762282e-01]	 2.0124865e-01


.. parsed-literal::

      28	 7.2933855e-01	 1.9729403e-01	 7.6673778e-01	 1.9532993e-01	[ 7.5301399e-01]	 2.1317172e-01


.. parsed-literal::

      29	 7.5243099e-01	 1.9611041e-01	 7.9046143e-01	 1.9274120e-01	[ 7.7630002e-01]	 2.4447727e-01


.. parsed-literal::

      30	 7.8087743e-01	 1.9807822e-01	 8.1971347e-01	 1.9203566e-01	[ 8.0719202e-01]	 2.1488500e-01


.. parsed-literal::

      31	 8.0922449e-01	 1.9421267e-01	 8.4879105e-01	 1.8788584e-01	[ 8.2644286e-01]	 2.0398808e-01


.. parsed-literal::

      32	 8.3924969e-01	 1.9070473e-01	 8.7953064e-01	 1.8444498e-01	[ 8.4939780e-01]	 2.1082497e-01


.. parsed-literal::

      33	 8.6736431e-01	 1.8745472e-01	 9.0910352e-01	 1.8223355e-01	[ 8.6302285e-01]	 2.0779324e-01


.. parsed-literal::

      34	 8.8872148e-01	 1.8805852e-01	 9.3106006e-01	 1.8081663e-01	[ 8.8897144e-01]	 2.0903826e-01
      35	 9.1403834e-01	 1.8433441e-01	 9.5727002e-01	 1.7819347e-01	[ 9.0325881e-01]	 1.8660593e-01


.. parsed-literal::

      36	 9.3300320e-01	 1.8196010e-01	 9.7598487e-01	 1.7581804e-01	[ 9.3002757e-01]	 2.1364546e-01


.. parsed-literal::

      37	 9.4681391e-01	 1.8037891e-01	 9.8993693e-01	 1.7449469e-01	[ 9.4452472e-01]	 2.1846247e-01


.. parsed-literal::

      38	 9.5828350e-01	 1.8030276e-01	 1.0016230e+00	 1.7483475e-01	[ 9.5695888e-01]	 2.1931219e-01


.. parsed-literal::

      39	 9.6880595e-01	 1.7911691e-01	 1.0121311e+00	 1.7381003e-01	[ 9.6327461e-01]	 2.1925783e-01


.. parsed-literal::

      40	 9.9031673e-01	 1.7667917e-01	 1.0348105e+00	 1.7165097e-01	[ 9.7213809e-01]	 2.1282053e-01


.. parsed-literal::

      41	 1.0012118e+00	 1.7547561e-01	 1.0468965e+00	 1.7072758e-01	[ 9.7673014e-01]	 2.1280622e-01
      42	 1.0147251e+00	 1.7466578e-01	 1.0601992e+00	 1.6951957e-01	[ 9.9018438e-01]	 1.7610550e-01


.. parsed-literal::

      43	 1.0257920e+00	 1.7445483e-01	 1.0715775e+00	 1.6887158e-01	[ 1.0025626e+00]	 2.1452236e-01


.. parsed-literal::

      44	 1.0345994e+00	 1.7327060e-01	 1.0805920e+00	 1.6749008e-01	[ 1.0118352e+00]	 2.0587254e-01


.. parsed-literal::

      45	 1.0522835e+00	 1.6841095e-01	 1.0985974e+00	 1.6262119e-01	[ 1.0356155e+00]	 2.0243025e-01


.. parsed-literal::

      46	 1.0594544e+00	 1.6702337e-01	 1.1058402e+00	 1.6260315e-01	  1.0300149e+00 	 2.1919155e-01
      47	 1.0695565e+00	 1.6556741e-01	 1.1153214e+00	 1.6091446e-01	[ 1.0373349e+00]	 1.9264460e-01


.. parsed-literal::

      48	 1.0771067e+00	 1.6481722e-01	 1.1231215e+00	 1.5979701e-01	[ 1.0430648e+00]	 2.0281315e-01


.. parsed-literal::

      49	 1.0852213e+00	 1.6404523e-01	 1.1315508e+00	 1.5892476e-01	  1.0430435e+00 	 2.0986843e-01


.. parsed-literal::

      50	 1.0954173e+00	 1.6358758e-01	 1.1423467e+00	 1.5842227e-01	  1.0259278e+00 	 2.1468735e-01


.. parsed-literal::

      51	 1.1039951e+00	 1.6316484e-01	 1.1508762e+00	 1.5771993e-01	  1.0320984e+00 	 2.1162677e-01


.. parsed-literal::

      52	 1.1090463e+00	 1.6274486e-01	 1.1558248e+00	 1.5742977e-01	  1.0356296e+00 	 2.0636034e-01


.. parsed-literal::

      53	 1.1187175e+00	 1.6208078e-01	 1.1656814e+00	 1.5683621e-01	  1.0418474e+00 	 2.0393634e-01


.. parsed-literal::

      54	 1.1280869e+00	 1.6131158e-01	 1.1756522e+00	 1.5605917e-01	  1.0335489e+00 	 2.0803070e-01


.. parsed-literal::

      55	 1.1384101e+00	 1.5987920e-01	 1.1862436e+00	 1.5460750e-01	  1.0333983e+00 	 2.1374106e-01


.. parsed-literal::

      56	 1.1478202e+00	 1.5823536e-01	 1.1959030e+00	 1.5268045e-01	  1.0346158e+00 	 2.0921183e-01
      57	 1.1569244e+00	 1.5712758e-01	 1.2052945e+00	 1.5110910e-01	  1.0326608e+00 	 1.8789577e-01


.. parsed-literal::

      58	 1.1594250e+00	 1.5706580e-01	 1.2085852e+00	 1.4965904e-01	  1.0293968e+00 	 1.8452406e-01
      59	 1.1722037e+00	 1.5597059e-01	 1.2211055e+00	 1.4887542e-01	  1.0341038e+00 	 1.8246245e-01


.. parsed-literal::

      60	 1.1763865e+00	 1.5571659e-01	 1.2251030e+00	 1.4873952e-01	  1.0425951e+00 	 2.0971084e-01


.. parsed-literal::

      61	 1.1828950e+00	 1.5520725e-01	 1.2317948e+00	 1.4809763e-01	  1.0413103e+00 	 2.1262908e-01


.. parsed-literal::

      62	 1.1911713e+00	 1.5461660e-01	 1.2402204e+00	 1.4718351e-01	[ 1.0455768e+00]	 2.0866370e-01


.. parsed-literal::

      63	 1.1980853e+00	 1.5347857e-01	 1.2474802e+00	 1.4593715e-01	  1.0397347e+00 	 2.1031380e-01


.. parsed-literal::

      64	 1.2044840e+00	 1.5278271e-01	 1.2538535e+00	 1.4538357e-01	  1.0399542e+00 	 2.0367861e-01


.. parsed-literal::

      65	 1.2125251e+00	 1.5190828e-01	 1.2619800e+00	 1.4466585e-01	[ 1.0534880e+00]	 2.0947671e-01


.. parsed-literal::

      66	 1.2192320e+00	 1.5016266e-01	 1.2693550e+00	 1.4356490e-01	  1.0484136e+00 	 2.0453572e-01
      67	 1.2263875e+00	 1.4993410e-01	 1.2763347e+00	 1.4319490e-01	[ 1.0789063e+00]	 1.8213344e-01


.. parsed-literal::

      68	 1.2301726e+00	 1.4981320e-01	 1.2800953e+00	 1.4304609e-01	[ 1.0910007e+00]	 2.1032238e-01


.. parsed-literal::

      69	 1.2371465e+00	 1.4926142e-01	 1.2873455e+00	 1.4245707e-01	[ 1.1000236e+00]	 2.0203352e-01
      70	 1.2418034e+00	 1.4888486e-01	 1.2923694e+00	 1.4231288e-01	[ 1.1034914e+00]	 2.0841265e-01


.. parsed-literal::

      71	 1.2484801e+00	 1.4809033e-01	 1.2990293e+00	 1.4159545e-01	  1.1013001e+00 	 1.9604659e-01


.. parsed-literal::

      72	 1.2534962e+00	 1.4740785e-01	 1.3041297e+00	 1.4115052e-01	  1.0927249e+00 	 2.1256089e-01


.. parsed-literal::

      73	 1.2580630e+00	 1.4702359e-01	 1.3087716e+00	 1.4096836e-01	  1.0853630e+00 	 2.1416783e-01


.. parsed-literal::

      74	 1.2690185e+00	 1.4640725e-01	 1.3200943e+00	 1.4087304e-01	  1.0574354e+00 	 2.1168470e-01


.. parsed-literal::

      75	 1.2735186e+00	 1.4640064e-01	 1.3246868e+00	 1.4115236e-01	  1.0373247e+00 	 3.1906700e-01


.. parsed-literal::

      76	 1.2794128e+00	 1.4588706e-01	 1.3306685e+00	 1.4096289e-01	  1.0249555e+00 	 2.0402098e-01


.. parsed-literal::

      77	 1.2840465e+00	 1.4538949e-01	 1.3353248e+00	 1.4060315e-01	  1.0231777e+00 	 2.1243167e-01


.. parsed-literal::

      78	 1.2884383e+00	 1.4465679e-01	 1.3398556e+00	 1.3986422e-01	  1.0250555e+00 	 2.1084166e-01
      79	 1.2932854e+00	 1.4428840e-01	 1.3447293e+00	 1.3966984e-01	  1.0207085e+00 	 1.8960714e-01


.. parsed-literal::

      80	 1.2997447e+00	 1.4381370e-01	 1.3513926e+00	 1.3932564e-01	  1.0081693e+00 	 2.0601416e-01
      81	 1.3060295e+00	 1.4342950e-01	 1.3578513e+00	 1.3903147e-01	  1.0005527e+00 	 1.9805312e-01


.. parsed-literal::

      82	 1.3108842e+00	 1.4353627e-01	 1.3628333e+00	 1.3909292e-01	  9.8727139e-01 	 2.0152640e-01
      83	 1.3153285e+00	 1.4337267e-01	 1.3671773e+00	 1.3888142e-01	  9.9849036e-01 	 1.9047117e-01


.. parsed-literal::

      84	 1.3196882e+00	 1.4332346e-01	 1.3715437e+00	 1.3883984e-01	  1.0065971e+00 	 2.0314169e-01
      85	 1.3251064e+00	 1.4330707e-01	 1.3771621e+00	 1.3899982e-01	  1.0087522e+00 	 1.8791318e-01


.. parsed-literal::

      86	 1.3281254e+00	 1.4378545e-01	 1.3805403e+00	 1.3973591e-01	  1.0307083e+00 	 2.1448326e-01


.. parsed-literal::

      87	 1.3357355e+00	 1.4347395e-01	 1.3881249e+00	 1.3966838e-01	  1.0296868e+00 	 2.1183372e-01


.. parsed-literal::

      88	 1.3387301e+00	 1.4321008e-01	 1.3911012e+00	 1.3955047e-01	  1.0294106e+00 	 2.0516849e-01


.. parsed-literal::

      89	 1.3445659e+00	 1.4275465e-01	 1.3971034e+00	 1.3949639e-01	  1.0282586e+00 	 2.1057415e-01


.. parsed-literal::

      90	 1.3514917e+00	 1.4252622e-01	 1.4042021e+00	 1.3985707e-01	  1.0281457e+00 	 2.1157956e-01


.. parsed-literal::

      91	 1.3534671e+00	 1.4250715e-01	 1.4068187e+00	 1.4048365e-01	  1.0285653e+00 	 2.0737791e-01


.. parsed-literal::

      92	 1.3605304e+00	 1.4246764e-01	 1.4134286e+00	 1.4036602e-01	  1.0352157e+00 	 2.0880699e-01


.. parsed-literal::

      93	 1.3629832e+00	 1.4254943e-01	 1.4158553e+00	 1.4053733e-01	  1.0355243e+00 	 2.1014500e-01


.. parsed-literal::

      94	 1.3676385e+00	 1.4302872e-01	 1.4207368e+00	 1.4129497e-01	  1.0211054e+00 	 2.1165967e-01
      95	 1.3715005e+00	 1.4305159e-01	 1.4247366e+00	 1.4163478e-01	  1.0090973e+00 	 1.9970989e-01


.. parsed-literal::

      96	 1.3747914e+00	 1.4294440e-01	 1.4280770e+00	 1.4165963e-01	  1.0045236e+00 	 2.0335078e-01


.. parsed-literal::

      97	 1.3805162e+00	 1.4281045e-01	 1.4340022e+00	 1.4185538e-01	  9.8911407e-01 	 2.0726395e-01


.. parsed-literal::

      98	 1.3838847e+00	 1.4294750e-01	 1.4374181e+00	 1.4226445e-01	  9.8526458e-01 	 2.1099615e-01
      99	 1.3885564e+00	 1.4370227e-01	 1.4422386e+00	 1.4335064e-01	  9.7770869e-01 	 1.9757318e-01


.. parsed-literal::

     100	 1.3927207e+00	 1.4364643e-01	 1.4462663e+00	 1.4356133e-01	  9.8867696e-01 	 1.8783355e-01
     101	 1.3953436e+00	 1.4361152e-01	 1.4488219e+00	 1.4357610e-01	  9.9605010e-01 	 1.8433666e-01


.. parsed-literal::

     102	 1.4021477e+00	 1.4340656e-01	 1.4556885e+00	 1.4345958e-01	  1.0093355e+00 	 2.0592546e-01


.. parsed-literal::

     103	 1.4052750e+00	 1.4307785e-01	 1.4588235e+00	 1.4338865e-01	  1.0131147e+00 	 2.8391147e-01


.. parsed-literal::

     104	 1.4090122e+00	 1.4282166e-01	 1.4625698e+00	 1.4306757e-01	  1.0258321e+00 	 2.1533012e-01
     105	 1.4133620e+00	 1.4225817e-01	 1.4669221e+00	 1.4251454e-01	  1.0339288e+00 	 2.0278049e-01


.. parsed-literal::

     106	 1.4167398e+00	 1.4204956e-01	 1.4704169e+00	 1.4233817e-01	  1.0515278e+00 	 2.0778298e-01


.. parsed-literal::

     107	 1.4204295e+00	 1.4152044e-01	 1.4740039e+00	 1.4181746e-01	  1.0517268e+00 	 2.0908618e-01


.. parsed-literal::

     108	 1.4237516e+00	 1.4128121e-01	 1.4772933e+00	 1.4169030e-01	  1.0516575e+00 	 2.1339941e-01


.. parsed-literal::

     109	 1.4270696e+00	 1.4089322e-01	 1.4806471e+00	 1.4134773e-01	  1.0533089e+00 	 2.1190333e-01


.. parsed-literal::

     110	 1.4312108e+00	 1.4076811e-01	 1.4848327e+00	 1.4126726e-01	  1.0539525e+00 	 2.0872593e-01


.. parsed-literal::

     111	 1.4350751e+00	 1.4041240e-01	 1.4887381e+00	 1.4089865e-01	  1.0554809e+00 	 2.0510674e-01


.. parsed-literal::

     112	 1.4387488e+00	 1.3995882e-01	 1.4924659e+00	 1.4026252e-01	  1.0511921e+00 	 2.1483493e-01


.. parsed-literal::

     113	 1.4416026e+00	 1.3974537e-01	 1.4953050e+00	 1.4012863e-01	  1.0488636e+00 	 2.0879722e-01


.. parsed-literal::

     114	 1.4438165e+00	 1.3953070e-01	 1.4974752e+00	 1.3990477e-01	  1.0493333e+00 	 2.0784616e-01


.. parsed-literal::

     115	 1.4480954e+00	 1.3891664e-01	 1.5018045e+00	 1.3920802e-01	  1.0415860e+00 	 2.1149540e-01


.. parsed-literal::

     116	 1.4510433e+00	 1.3829651e-01	 1.5048974e+00	 1.3861221e-01	  1.0299363e+00 	 2.1983671e-01
     117	 1.4543749e+00	 1.3789967e-01	 1.5083496e+00	 1.3810489e-01	  1.0260824e+00 	 2.0521545e-01


.. parsed-literal::

     118	 1.4575745e+00	 1.3731185e-01	 1.5116746e+00	 1.3725140e-01	  1.0171071e+00 	 2.1244383e-01


.. parsed-literal::

     119	 1.4605592e+00	 1.3724710e-01	 1.5147235e+00	 1.3708149e-01	  1.0227008e+00 	 2.1847725e-01


.. parsed-literal::

     120	 1.4633892e+00	 1.3707590e-01	 1.5175448e+00	 1.3683426e-01	  1.0245559e+00 	 2.1578670e-01
     121	 1.4659342e+00	 1.3678645e-01	 1.5202075e+00	 1.3637358e-01	  1.0231489e+00 	 1.9297814e-01


.. parsed-literal::

     122	 1.4684039e+00	 1.3647847e-01	 1.5226524e+00	 1.3604494e-01	  1.0176794e+00 	 2.0153856e-01


.. parsed-literal::

     123	 1.4700573e+00	 1.3628414e-01	 1.5243144e+00	 1.3587973e-01	  1.0119818e+00 	 2.1937442e-01
     124	 1.4725000e+00	 1.3567338e-01	 1.5268719e+00	 1.3525336e-01	  1.0016761e+00 	 1.7875051e-01


.. parsed-literal::

     125	 1.4738245e+00	 1.3528239e-01	 1.5284301e+00	 1.3483447e-01	  9.8254305e-01 	 2.1532631e-01


.. parsed-literal::

     126	 1.4759077e+00	 1.3526328e-01	 1.5303956e+00	 1.3485277e-01	  9.9199341e-01 	 2.1537161e-01
     127	 1.4774315e+00	 1.3505839e-01	 1.5319398e+00	 1.3466136e-01	  9.9531433e-01 	 2.0731044e-01


.. parsed-literal::

     128	 1.4792908e+00	 1.3474073e-01	 1.5338446e+00	 1.3439326e-01	  9.9647625e-01 	 2.1471000e-01


.. parsed-literal::

     129	 1.4829958e+00	 1.3416770e-01	 1.5376478e+00	 1.3416759e-01	  9.9635324e-01 	 2.0455027e-01


.. parsed-literal::

     130	 1.4851596e+00	 1.3350339e-01	 1.5398475e+00	 1.3359457e-01	  9.9022451e-01 	 3.1049895e-01


.. parsed-literal::

     131	 1.4871620e+00	 1.3335896e-01	 1.5418311e+00	 1.3361730e-01	  9.8702960e-01 	 2.1476793e-01


.. parsed-literal::

     132	 1.4887521e+00	 1.3323468e-01	 1.5433698e+00	 1.3355691e-01	  9.8101931e-01 	 2.1227288e-01


.. parsed-literal::

     133	 1.4902665e+00	 1.3311278e-01	 1.5448462e+00	 1.3338683e-01	  9.7704642e-01 	 2.1517992e-01
     134	 1.4924065e+00	 1.3277510e-01	 1.5469819e+00	 1.3304388e-01	  9.6911641e-01 	 1.7964411e-01


.. parsed-literal::

     135	 1.4950512e+00	 1.3232248e-01	 1.5497047e+00	 1.3252018e-01	  9.6204616e-01 	 2.2144175e-01


.. parsed-literal::

     136	 1.4957714e+00	 1.3144690e-01	 1.5506963e+00	 1.3154577e-01	  9.3745778e-01 	 2.1645880e-01


.. parsed-literal::

     137	 1.4984846e+00	 1.3169738e-01	 1.5532797e+00	 1.3174647e-01	  9.5069123e-01 	 2.1562386e-01


.. parsed-literal::

     138	 1.4993624e+00	 1.3165817e-01	 1.5541521e+00	 1.3172610e-01	  9.5247149e-01 	 2.0815635e-01
     139	 1.5011580e+00	 1.3143493e-01	 1.5559645e+00	 1.3151227e-01	  9.5087452e-01 	 1.8715549e-01


.. parsed-literal::

     140	 1.5031144e+00	 1.3094278e-01	 1.5579083e+00	 1.3118907e-01	  9.5563351e-01 	 2.1301651e-01


.. parsed-literal::

     141	 1.5037558e+00	 1.2997859e-01	 1.5586747e+00	 1.3035621e-01	  9.4467028e-01 	 2.0854902e-01


.. parsed-literal::

     142	 1.5065747e+00	 1.3009136e-01	 1.5613269e+00	 1.3045383e-01	  9.6130994e-01 	 2.0446873e-01
     143	 1.5075078e+00	 1.3002958e-01	 1.5622284e+00	 1.3039422e-01	  9.6545627e-01 	 1.8561888e-01


.. parsed-literal::

     144	 1.5092577e+00	 1.2980126e-01	 1.5639679e+00	 1.3013822e-01	  9.7204905e-01 	 1.8296385e-01


.. parsed-literal::

     145	 1.5104688e+00	 1.2937135e-01	 1.5653441e+00	 1.2987151e-01	  9.7336071e-01 	 2.2371411e-01


.. parsed-literal::

     146	 1.5126259e+00	 1.2929272e-01	 1.5674393e+00	 1.2978210e-01	  9.7416941e-01 	 2.0862484e-01


.. parsed-literal::

     147	 1.5141510e+00	 1.2916517e-01	 1.5689952e+00	 1.2970713e-01	  9.7072695e-01 	 2.1090102e-01
     148	 1.5158247e+00	 1.2900659e-01	 1.5707326e+00	 1.2971280e-01	  9.6337829e-01 	 2.0006776e-01


.. parsed-literal::

     149	 1.5170220e+00	 1.2894192e-01	 1.5720544e+00	 1.2976159e-01	  9.5693789e-01 	 1.8724799e-01
     150	 1.5193595e+00	 1.2867610e-01	 1.5743420e+00	 1.2975927e-01	  9.4963550e-01 	 1.8475938e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.28 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fdf244d9720>



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
    CPU times: user 1.77 s, sys: 41 ms, total: 1.81 s
    Wall time: 597 ms


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

