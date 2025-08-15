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
       1	-3.5387227e-01	 3.2381774e-01	-3.4422485e-01	 3.0740120e-01	[-3.1469957e-01]	 4.6832108e-01


.. parsed-literal::

       2	-2.8233653e-01	 3.1284255e-01	-2.5810054e-01	 2.9882390e-01	[-2.1729716e-01]	 2.3384571e-01


.. parsed-literal::

       3	-2.3937942e-01	 2.9275892e-01	-1.9777263e-01	 2.8191441e-01	[-1.5613305e-01]	 2.8002739e-01


.. parsed-literal::

       4	-1.9841067e-01	 2.6800666e-01	-1.5582667e-01	 2.6111507e-01	[-1.1781817e-01]	 2.1946692e-01
       5	-1.1209431e-01	 2.5837551e-01	-7.8213978e-02	 2.5218096e-01	[-5.0517866e-02]	 1.9721937e-01


.. parsed-literal::

       6	-7.5269581e-02	 2.5289498e-01	-4.5103861e-02	 2.4802309e-01	[-2.6473648e-02]	 2.1727729e-01


.. parsed-literal::

       7	-5.8716810e-02	 2.5026627e-01	-3.4072374e-02	 2.4520074e-01	[-1.3800955e-02]	 2.0330071e-01


.. parsed-literal::

       8	-4.5379682e-02	 2.4805105e-01	-2.4906024e-02	 2.4271131e-01	[-3.2243551e-03]	 2.1193695e-01
       9	-3.3014966e-02	 2.4580078e-01	-1.5319591e-02	 2.4046505e-01	[ 6.3090657e-03]	 1.8080854e-01


.. parsed-literal::

      10	-2.1602088e-02	 2.4356960e-01	-6.0499619e-03	 2.3763832e-01	[ 1.9627339e-02]	 2.0365238e-01


.. parsed-literal::

      11	-1.7793630e-02	 2.4307283e-01	-3.6489073e-03	 2.3731811e-01	  1.8809129e-02 	 2.1308351e-01


.. parsed-literal::

      12	-1.3827787e-02	 2.4244831e-01	 3.9680170e-05	 2.3671344e-01	[ 2.4592395e-02]	 2.0608878e-01
      13	-1.0727389e-02	 2.4177674e-01	 3.0878678e-03	 2.3586096e-01	[ 2.8673906e-02]	 1.9993067e-01


.. parsed-literal::

      14	-6.3100009e-03	 2.4078696e-01	 8.0071547e-03	 2.3495692e-01	[ 3.4113767e-02]	 2.0795250e-01


.. parsed-literal::

      15	 1.4429546e-01	 2.2489677e-01	 1.6666812e-01	 2.1764899e-01	[ 1.8989892e-01]	 3.2887912e-01
      16	 1.8097724e-01	 2.2129237e-01	 2.0473095e-01	 2.1716008e-01	[ 2.1868413e-01]	 1.8644214e-01


.. parsed-literal::

      17	 2.8108860e-01	 2.1738054e-01	 3.0875082e-01	 2.1230969e-01	[ 3.2718353e-01]	 2.0902729e-01
      18	 3.2155488e-01	 2.1336091e-01	 3.5561709e-01	 2.0690458e-01	[ 3.8029348e-01]	 1.9693375e-01


.. parsed-literal::

      19	 3.7598604e-01	 2.0968306e-01	 4.0911092e-01	 2.0263716e-01	[ 4.2760920e-01]	 2.0471954e-01


.. parsed-literal::

      20	 4.1270527e-01	 2.0655312e-01	 4.4558967e-01	 2.0020657e-01	[ 4.6055194e-01]	 2.0950103e-01
      21	 4.5883917e-01	 2.0345705e-01	 4.9197197e-01	 1.9930159e-01	[ 5.0026474e-01]	 1.9577789e-01


.. parsed-literal::

      22	 5.6059941e-01	 2.0089292e-01	 5.9509288e-01	 1.9884232e-01	[ 5.9352569e-01]	 2.1840096e-01


.. parsed-literal::

      23	 5.7424854e-01	 2.0844871e-01	 6.1583485e-01	 2.0459952e-01	[ 6.0177081e-01]	 2.1199393e-01
      24	 6.4619238e-01	 1.9950735e-01	 6.8355123e-01	 1.9661081e-01	[ 6.7100340e-01]	 1.8864417e-01


.. parsed-literal::

      25	 6.7148052e-01	 1.9792896e-01	 7.0819703e-01	 1.9407018e-01	[ 6.9903304e-01]	 2.1477771e-01


.. parsed-literal::

      26	 6.9380829e-01	 1.9854688e-01	 7.3085782e-01	 1.9363177e-01	[ 7.2278442e-01]	 2.0877552e-01


.. parsed-literal::

      27	 7.1559008e-01	 2.0242583e-01	 7.5233644e-01	 1.9429116e-01	[ 7.4983922e-01]	 2.2068048e-01
      28	 7.4144825e-01	 2.0467531e-01	 7.7887180e-01	 1.9495769e-01	[ 7.7234486e-01]	 1.8736815e-01


.. parsed-literal::

      29	 7.5514784e-01	 1.9786314e-01	 7.9282077e-01	 1.9050507e-01	[ 7.8475733e-01]	 1.9489408e-01


.. parsed-literal::

      30	 7.7509205e-01	 1.9732335e-01	 8.1254215e-01	 1.8953865e-01	[ 8.0466850e-01]	 2.0934153e-01


.. parsed-literal::

      31	 8.0064977e-01	 1.9751404e-01	 8.3827832e-01	 1.8864887e-01	[ 8.3205907e-01]	 3.2574749e-01
      32	 8.3072928e-01	 1.9368854e-01	 8.6958634e-01	 1.8635619e-01	[ 8.6921558e-01]	 1.8953395e-01


.. parsed-literal::

      33	 8.5862308e-01	 1.9258251e-01	 8.9881213e-01	 1.8476005e-01	[ 9.0493719e-01]	 2.0993185e-01
      34	 8.7901787e-01	 1.9059698e-01	 9.2026622e-01	 1.8267900e-01	[ 9.2893836e-01]	 1.9072556e-01


.. parsed-literal::

      35	 9.0040294e-01	 1.8868701e-01	 9.4245378e-01	 1.8134164e-01	[ 9.5097118e-01]	 2.0539689e-01


.. parsed-literal::

      36	 9.1887770e-01	 1.8780020e-01	 9.6103156e-01	 1.7848062e-01	[ 9.7417055e-01]	 2.1142697e-01


.. parsed-literal::

      37	 9.3323688e-01	 1.8599348e-01	 9.7537728e-01	 1.7687591e-01	[ 9.8690754e-01]	 2.1135616e-01


.. parsed-literal::

      38	 9.5324179e-01	 1.8338836e-01	 9.9582500e-01	 1.7504195e-01	[ 1.0030541e+00]	 2.1970749e-01


.. parsed-literal::

      39	 9.6324081e-01	 1.8268178e-01	 1.0062653e+00	 1.7673905e-01	[ 1.0033375e+00]	 2.0958996e-01


.. parsed-literal::

      40	 9.7564459e-01	 1.8129458e-01	 1.0188001e+00	 1.7516632e-01	[ 1.0193993e+00]	 2.0646811e-01
      41	 9.8524475e-01	 1.8098275e-01	 1.0287360e+00	 1.7462204e-01	[ 1.0318458e+00]	 1.8222547e-01


.. parsed-literal::

      42	 9.9545478e-01	 1.8037470e-01	 1.0394167e+00	 1.7474522e-01	[ 1.0382818e+00]	 2.0393467e-01


.. parsed-literal::

      43	 1.0082655e+00	 1.8109919e-01	 1.0530342e+00	 1.7577876e-01	[ 1.0478577e+00]	 2.0967674e-01


.. parsed-literal::

      44	 1.0186696e+00	 1.8019010e-01	 1.0633659e+00	 1.7539109e-01	[ 1.0511088e+00]	 2.0808887e-01


.. parsed-literal::

      45	 1.0293241e+00	 1.7946637e-01	 1.0741010e+00	 1.7427079e-01	[ 1.0561799e+00]	 2.0979524e-01


.. parsed-literal::

      46	 1.0400570e+00	 1.7879252e-01	 1.0850410e+00	 1.7242111e-01	[ 1.0640399e+00]	 2.1440840e-01


.. parsed-literal::

      47	 1.0514120e+00	 1.7763228e-01	 1.0970508e+00	 1.6945364e-01	[ 1.0701541e+00]	 2.1684813e-01


.. parsed-literal::

      48	 1.0615268e+00	 1.7578885e-01	 1.1074546e+00	 1.6855207e-01	[ 1.0749705e+00]	 2.1253395e-01


.. parsed-literal::

      49	 1.0701980e+00	 1.7422332e-01	 1.1160440e+00	 1.6766395e-01	[ 1.0856530e+00]	 2.1167660e-01


.. parsed-literal::

      50	 1.0779879e+00	 1.7251500e-01	 1.1240454e+00	 1.6679186e-01	[ 1.0925926e+00]	 2.2144055e-01


.. parsed-literal::

      51	 1.0893355e+00	 1.7032796e-01	 1.1356945e+00	 1.6527406e-01	[ 1.1003574e+00]	 2.1381879e-01


.. parsed-literal::

      52	 1.0994309e+00	 1.6720647e-01	 1.1461100e+00	 1.6359256e-01	[ 1.1101402e+00]	 2.1590400e-01


.. parsed-literal::

      53	 1.1089801e+00	 1.6609500e-01	 1.1556503e+00	 1.6298603e-01	[ 1.1114807e+00]	 2.1621752e-01


.. parsed-literal::

      54	 1.1146100e+00	 1.6566447e-01	 1.1611005e+00	 1.6239589e-01	[ 1.1177209e+00]	 2.1947837e-01


.. parsed-literal::

      55	 1.1226829e+00	 1.6449270e-01	 1.1691373e+00	 1.6178188e-01	[ 1.1223925e+00]	 2.1319509e-01
      56	 1.1321765e+00	 1.6210975e-01	 1.1789563e+00	 1.5991214e-01	[ 1.1333272e+00]	 1.9192934e-01


.. parsed-literal::

      57	 1.1407846e+00	 1.6020499e-01	 1.1878250e+00	 1.5958505e-01	  1.1317205e+00 	 2.1351433e-01
      58	 1.1480992e+00	 1.5867763e-01	 1.1952824e+00	 1.5843863e-01	[ 1.1382773e+00]	 1.9524002e-01


.. parsed-literal::

      59	 1.1643706e+00	 1.5500061e-01	 1.2124122e+00	 1.5552929e-01	[ 1.1415727e+00]	 1.9665885e-01
      60	 1.1669985e+00	 1.5326924e-01	 1.2157456e+00	 1.5273916e-01	[ 1.1513913e+00]	 1.7303157e-01


.. parsed-literal::

      61	 1.1779027e+00	 1.5262509e-01	 1.2262330e+00	 1.5301365e-01	[ 1.1560039e+00]	 2.0573187e-01


.. parsed-literal::

      62	 1.1835489e+00	 1.5217045e-01	 1.2319241e+00	 1.5290420e-01	[ 1.1560155e+00]	 2.1967435e-01


.. parsed-literal::

      63	 1.1901232e+00	 1.5134754e-01	 1.2387696e+00	 1.5222834e-01	[ 1.1564250e+00]	 2.1615505e-01


.. parsed-literal::

      64	 1.1988962e+00	 1.5025235e-01	 1.2481166e+00	 1.5092435e-01	[ 1.1647796e+00]	 2.2576499e-01
      65	 1.2015828e+00	 1.4873181e-01	 1.2518521e+00	 1.4954115e-01	  1.1528185e+00 	 2.0528960e-01


.. parsed-literal::

      66	 1.2091798e+00	 1.4866025e-01	 1.2589835e+00	 1.4933810e-01	[ 1.1699914e+00]	 2.0092487e-01
      67	 1.2121776e+00	 1.4854166e-01	 1.2618762e+00	 1.4933387e-01	[ 1.1744523e+00]	 1.9845843e-01


.. parsed-literal::

      68	 1.2190059e+00	 1.4775459e-01	 1.2687770e+00	 1.4869771e-01	[ 1.1827524e+00]	 2.0702815e-01


.. parsed-literal::

      69	 1.2269106e+00	 1.4691941e-01	 1.2768768e+00	 1.4879113e-01	  1.1795130e+00 	 2.1489549e-01


.. parsed-literal::

      70	 1.2358954e+00	 1.4547897e-01	 1.2861381e+00	 1.4768456e-01	[ 1.1844334e+00]	 2.1862602e-01


.. parsed-literal::

      71	 1.2428368e+00	 1.4409911e-01	 1.2931814e+00	 1.4697755e-01	  1.1833635e+00 	 2.0920920e-01
      72	 1.2485873e+00	 1.4319863e-01	 1.2989316e+00	 1.4616179e-01	[ 1.1869558e+00]	 1.8652058e-01


.. parsed-literal::

      73	 1.2552766e+00	 1.4199793e-01	 1.3056296e+00	 1.4510563e-01	[ 1.1947113e+00]	 2.1358895e-01


.. parsed-literal::

      74	 1.2621384e+00	 1.4111864e-01	 1.3128011e+00	 1.4363580e-01	[ 1.2091425e+00]	 2.1116519e-01


.. parsed-literal::

      75	 1.2673936e+00	 1.4077996e-01	 1.3180249e+00	 1.4304134e-01	[ 1.2191661e+00]	 2.1752095e-01


.. parsed-literal::

      76	 1.2717320e+00	 1.4047254e-01	 1.3223896e+00	 1.4302592e-01	[ 1.2238900e+00]	 2.1065712e-01
      77	 1.2762657e+00	 1.3993813e-01	 1.3270839e+00	 1.4268059e-01	[ 1.2293846e+00]	 1.7915201e-01


.. parsed-literal::

      78	 1.2808100e+00	 1.3915202e-01	 1.3317278e+00	 1.4231099e-01	  1.2282335e+00 	 2.1005702e-01


.. parsed-literal::

      79	 1.2858448e+00	 1.3829641e-01	 1.3368558e+00	 1.4171250e-01	[ 1.2327366e+00]	 2.0287275e-01


.. parsed-literal::

      80	 1.2915964e+00	 1.3746813e-01	 1.3426608e+00	 1.4118594e-01	[ 1.2355444e+00]	 2.1150494e-01


.. parsed-literal::

      81	 1.2969528e+00	 1.3680217e-01	 1.3481373e+00	 1.4065111e-01	[ 1.2359826e+00]	 2.1192598e-01
      82	 1.3016550e+00	 1.3663126e-01	 1.3531240e+00	 1.4093592e-01	  1.2330253e+00 	 1.8929601e-01


.. parsed-literal::

      83	 1.3071014e+00	 1.3621295e-01	 1.3584349e+00	 1.4053684e-01	[ 1.2374078e+00]	 2.1957970e-01


.. parsed-literal::

      84	 1.3109692e+00	 1.3590042e-01	 1.3623654e+00	 1.4014292e-01	[ 1.2398945e+00]	 2.0787239e-01


.. parsed-literal::

      85	 1.3152715e+00	 1.3574785e-01	 1.3667020e+00	 1.3990375e-01	  1.2393604e+00 	 2.0747662e-01


.. parsed-literal::

      86	 1.3215019e+00	 1.3545744e-01	 1.3730372e+00	 1.3933456e-01	  1.2391904e+00 	 2.1674967e-01
      87	 1.3260806e+00	 1.3562042e-01	 1.3776867e+00	 1.3956450e-01	  1.2301005e+00 	 1.8034458e-01


.. parsed-literal::

      88	 1.3302926e+00	 1.3541021e-01	 1.3817459e+00	 1.3931719e-01	  1.2380552e+00 	 1.6905236e-01
      89	 1.3353444e+00	 1.3494649e-01	 1.3868398e+00	 1.3887082e-01	  1.2372660e+00 	 1.7135501e-01


.. parsed-literal::

      90	 1.3391092e+00	 1.3496950e-01	 1.3907543e+00	 1.3910954e-01	  1.2333798e+00 	 2.0176387e-01


.. parsed-literal::

      91	 1.3439548e+00	 1.3477126e-01	 1.3958362e+00	 1.3871848e-01	  1.2253189e+00 	 2.1874428e-01


.. parsed-literal::

      92	 1.3492595e+00	 1.3434548e-01	 1.4013426e+00	 1.3850215e-01	  1.2160867e+00 	 2.1085191e-01


.. parsed-literal::

      93	 1.3541260e+00	 1.3419527e-01	 1.4063571e+00	 1.3840462e-01	  1.2142789e+00 	 2.1203899e-01


.. parsed-literal::

      94	 1.3583220e+00	 1.3397229e-01	 1.4105790e+00	 1.3859888e-01	  1.2150303e+00 	 2.1202374e-01


.. parsed-literal::

      95	 1.3620857e+00	 1.3378979e-01	 1.4143845e+00	 1.3886370e-01	  1.2142064e+00 	 2.1313596e-01


.. parsed-literal::

      96	 1.3659251e+00	 1.3380991e-01	 1.4183120e+00	 1.3927965e-01	  1.2089552e+00 	 2.0677590e-01


.. parsed-literal::

      97	 1.3707334e+00	 1.3358064e-01	 1.4232299e+00	 1.3934148e-01	  1.2025579e+00 	 2.1562052e-01


.. parsed-literal::

      98	 1.3751064e+00	 1.3379957e-01	 1.4278085e+00	 1.3979476e-01	  1.1915293e+00 	 2.0743418e-01


.. parsed-literal::

      99	 1.3798056e+00	 1.3351047e-01	 1.4324614e+00	 1.3917434e-01	  1.1969113e+00 	 2.1046972e-01


.. parsed-literal::

     100	 1.3839698e+00	 1.3311899e-01	 1.4366570e+00	 1.3855397e-01	  1.1973951e+00 	 2.0469952e-01
     101	 1.3880742e+00	 1.3274451e-01	 1.4407563e+00	 1.3794646e-01	  1.1991172e+00 	 1.8341112e-01


.. parsed-literal::

     102	 1.3921864e+00	 1.3224205e-01	 1.4449388e+00	 1.3730275e-01	  1.1996828e+00 	 2.1013451e-01
     103	 1.3963759e+00	 1.3202517e-01	 1.4490697e+00	 1.3722707e-01	  1.1971207e+00 	 1.8395352e-01


.. parsed-literal::

     104	 1.3996431e+00	 1.3209460e-01	 1.4523541e+00	 1.3731850e-01	  1.1946687e+00 	 1.8198180e-01


.. parsed-literal::

     105	 1.4032724e+00	 1.3206777e-01	 1.4560647e+00	 1.3726634e-01	  1.1913360e+00 	 2.0675755e-01


.. parsed-literal::

     106	 1.4081335e+00	 1.3194063e-01	 1.4611456e+00	 1.3771459e-01	  1.1839131e+00 	 2.0989585e-01


.. parsed-literal::

     107	 1.4133135e+00	 1.3142002e-01	 1.4664036e+00	 1.3711259e-01	  1.1795969e+00 	 2.1487093e-01
     108	 1.4168878e+00	 1.3093241e-01	 1.4699288e+00	 1.3676005e-01	  1.1849630e+00 	 2.0048547e-01


.. parsed-literal::

     109	 1.4206616e+00	 1.3049998e-01	 1.4737618e+00	 1.3656831e-01	  1.1905291e+00 	 1.9171739e-01


.. parsed-literal::

     110	 1.4239827e+00	 1.3036265e-01	 1.4772096e+00	 1.3668311e-01	  1.1881012e+00 	 2.0198274e-01
     111	 1.4272309e+00	 1.3039958e-01	 1.4804353e+00	 1.3670966e-01	  1.1938667e+00 	 1.9741607e-01


.. parsed-literal::

     112	 1.4321323e+00	 1.3045538e-01	 1.4854884e+00	 1.3656364e-01	  1.1946006e+00 	 2.1103740e-01


.. parsed-literal::

     113	 1.4350544e+00	 1.3030728e-01	 1.4884163e+00	 1.3636683e-01	  1.2108036e+00 	 2.1356058e-01


.. parsed-literal::

     114	 1.4378192e+00	 1.3009618e-01	 1.4911466e+00	 1.3606734e-01	  1.2147483e+00 	 2.2034216e-01
     115	 1.4429959e+00	 1.2962888e-01	 1.4964090e+00	 1.3546032e-01	  1.2220740e+00 	 2.0049596e-01


.. parsed-literal::

     116	 1.4451975e+00	 1.2918353e-01	 1.4986411e+00	 1.3523713e-01	  1.2204872e+00 	 2.2995567e-01


.. parsed-literal::

     117	 1.4476992e+00	 1.2894279e-01	 1.5012115e+00	 1.3521871e-01	  1.2198341e+00 	 2.0223927e-01


.. parsed-literal::

     118	 1.4512094e+00	 1.2848021e-01	 1.5049740e+00	 1.3502936e-01	  1.2125475e+00 	 2.1793389e-01


.. parsed-literal::

     119	 1.4538309e+00	 1.2807935e-01	 1.5076555e+00	 1.3498614e-01	  1.2126979e+00 	 2.1665406e-01


.. parsed-literal::

     120	 1.4558588e+00	 1.2804238e-01	 1.5096119e+00	 1.3485093e-01	  1.2173892e+00 	 2.0684505e-01


.. parsed-literal::

     121	 1.4597730e+00	 1.2768854e-01	 1.5134619e+00	 1.3447860e-01	  1.2240934e+00 	 2.0904589e-01


.. parsed-literal::

     122	 1.4618284e+00	 1.2762637e-01	 1.5155140e+00	 1.3450280e-01	  1.2304300e+00 	 2.1392393e-01
     123	 1.4641465e+00	 1.2739161e-01	 1.5178117e+00	 1.3441833e-01	  1.2310622e+00 	 1.9605184e-01


.. parsed-literal::

     124	 1.4667274e+00	 1.2715800e-01	 1.5204407e+00	 1.3441554e-01	  1.2290115e+00 	 1.9179797e-01


.. parsed-literal::

     125	 1.4684528e+00	 1.2711922e-01	 1.5222345e+00	 1.3435373e-01	  1.2260492e+00 	 2.0975852e-01


.. parsed-literal::

     126	 1.4710344e+00	 1.2715746e-01	 1.5248603e+00	 1.3430824e-01	  1.2259446e+00 	 2.2030926e-01


.. parsed-literal::

     127	 1.4725012e+00	 1.2724073e-01	 1.5265226e+00	 1.3427124e-01	  1.2184467e+00 	 2.1142745e-01


.. parsed-literal::

     128	 1.4761740e+00	 1.2718721e-01	 1.5300610e+00	 1.3424597e-01	  1.2279106e+00 	 2.1795368e-01


.. parsed-literal::

     129	 1.4774381e+00	 1.2711484e-01	 1.5312744e+00	 1.3424899e-01	  1.2321256e+00 	 2.1038842e-01
     130	 1.4799332e+00	 1.2701524e-01	 1.5337316e+00	 1.3427013e-01	  1.2354280e+00 	 1.8160868e-01


.. parsed-literal::

     131	 1.4831748e+00	 1.2692052e-01	 1.5369343e+00	 1.3429880e-01	  1.2365136e+00 	 2.1490645e-01


.. parsed-literal::

     132	 1.4849911e+00	 1.2696102e-01	 1.5388933e+00	 1.3456082e-01	  1.2278406e+00 	 2.0417309e-01


.. parsed-literal::

     133	 1.4885028e+00	 1.2690087e-01	 1.5423139e+00	 1.3445912e-01	  1.2303535e+00 	 2.0752406e-01


.. parsed-literal::

     134	 1.4901222e+00	 1.2689166e-01	 1.5439812e+00	 1.3447055e-01	  1.2280876e+00 	 2.0803428e-01
     135	 1.4919983e+00	 1.2704625e-01	 1.5459801e+00	 1.3469106e-01	  1.2220016e+00 	 1.8206382e-01


.. parsed-literal::

     136	 1.4940328e+00	 1.2687800e-01	 1.5481171e+00	 1.3486392e-01	  1.2190962e+00 	 2.1786380e-01
     137	 1.4958076e+00	 1.2683493e-01	 1.5498826e+00	 1.3500900e-01	  1.2211172e+00 	 1.7722321e-01


.. parsed-literal::

     138	 1.4980672e+00	 1.2659435e-01	 1.5521318e+00	 1.3508814e-01	  1.2207631e+00 	 1.9844079e-01
     139	 1.4995693e+00	 1.2669058e-01	 1.5536047e+00	 1.3515390e-01	  1.2256367e+00 	 1.8391514e-01


.. parsed-literal::

     140	 1.5010929e+00	 1.2658460e-01	 1.5550975e+00	 1.3502423e-01	  1.2250449e+00 	 2.0398736e-01


.. parsed-literal::

     141	 1.5044604e+00	 1.2634370e-01	 1.5585154e+00	 1.3474909e-01	  1.2181634e+00 	 2.1104121e-01


.. parsed-literal::

     142	 1.5062426e+00	 1.2626819e-01	 1.5603935e+00	 1.3482096e-01	  1.2141122e+00 	 2.1389222e-01
     143	 1.5086855e+00	 1.2620718e-01	 1.5630439e+00	 1.3506198e-01	  1.2072729e+00 	 1.8498778e-01


.. parsed-literal::

     144	 1.5104701e+00	 1.2620741e-01	 1.5649193e+00	 1.3512553e-01	  1.2017931e+00 	 1.8370295e-01


.. parsed-literal::

     145	 1.5122214e+00	 1.2615349e-01	 1.5666838e+00	 1.3519471e-01	  1.2006409e+00 	 2.0300150e-01


.. parsed-literal::

     146	 1.5142823e+00	 1.2598533e-01	 1.5688077e+00	 1.3514571e-01	  1.1961178e+00 	 2.1198344e-01
     147	 1.5167122e+00	 1.2571471e-01	 1.5713425e+00	 1.3504783e-01	  1.1886043e+00 	 1.9067907e-01


.. parsed-literal::

     148	 1.5183365e+00	 1.2564123e-01	 1.5730012e+00	 1.3493504e-01	  1.1874808e+00 	 2.0038438e-01


.. parsed-literal::

     149	 1.5196141e+00	 1.2560602e-01	 1.5742118e+00	 1.3480441e-01	  1.1874047e+00 	 2.0466614e-01


.. parsed-literal::

     150	 1.5214323e+00	 1.2558950e-01	 1.5760157e+00	 1.3470339e-01	  1.1805766e+00 	 2.1005225e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.16 s, total: 2min 5s
    Wall time: 31.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f328ccf37f0>



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


.. parsed-literal::

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
    CPU times: user 2.18 s, sys: 58.9 ms, total: 2.24 s
    Wall time: 713 ms


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

