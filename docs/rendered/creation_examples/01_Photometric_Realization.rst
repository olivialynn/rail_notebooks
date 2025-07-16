Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f3a97c48910>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.062253  0.034179  
    1      25.391064  0.013378  0.009072  
    2      24.304707  0.121011  0.070842  
    3      25.291103  0.007488  0.003951  
    4      25.096743  0.258556  0.216492  
    ...          ...       ...       ...  
    99995  24.737946  0.175304  0.121080  
    99996  24.224169  0.012483  0.010594  
    99997  25.613836  0.050570  0.031026  
    99998  25.274899  0.029544  0.015822  
    99999  25.699642  0.133115  0.094508  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.568246</td>
          <td>0.146866</td>
          <td>26.113494</td>
          <td>0.087144</td>
          <td>25.342589</td>
          <td>0.071931</td>
          <td>24.697577</td>
          <td>0.077751</td>
          <td>24.086192</td>
          <td>0.101931</td>
          <td>0.062253</td>
          <td>0.034179</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.263758</td>
          <td>1.995853</td>
          <td>27.997244</td>
          <td>0.468125</td>
          <td>26.558162</td>
          <td>0.128527</td>
          <td>26.338844</td>
          <td>0.171257</td>
          <td>25.759405</td>
          <td>0.195008</td>
          <td>25.313333</td>
          <td>0.288245</td>
          <td>0.013378</td>
          <td>0.009072</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.046728</td>
          <td>0.958013</td>
          <td>28.134575</td>
          <td>0.465850</td>
          <td>26.209144</td>
          <td>0.153303</td>
          <td>25.068692</td>
          <td>0.107726</td>
          <td>24.305697</td>
          <td>0.123428</td>
          <td>0.121011</td>
          <td>0.070842</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.173538</td>
          <td>0.217018</td>
          <td>26.214303</td>
          <td>0.153983</td>
          <td>25.497982</td>
          <td>0.156188</td>
          <td>25.792107</td>
          <td>0.420089</td>
          <td>0.007488</td>
          <td>0.003951</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.387342</td>
          <td>0.345225</td>
          <td>25.992474</td>
          <td>0.089012</td>
          <td>25.834232</td>
          <td>0.068096</td>
          <td>25.620688</td>
          <td>0.091928</td>
          <td>25.383982</td>
          <td>0.141623</td>
          <td>25.334482</td>
          <td>0.293209</td>
          <td>0.258556</td>
          <td>0.216492</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.835658</td>
          <td>0.486518</td>
          <td>26.332447</td>
          <td>0.119803</td>
          <td>25.443511</td>
          <td>0.048146</td>
          <td>25.056598</td>
          <td>0.055822</td>
          <td>24.988414</td>
          <td>0.100421</td>
          <td>24.708278</td>
          <td>0.174431</td>
          <td>0.175304</td>
          <td>0.121080</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.891890</td>
          <td>2.545271</td>
          <td>26.844625</td>
          <td>0.185858</td>
          <td>25.977176</td>
          <td>0.077273</td>
          <td>25.127155</td>
          <td>0.059430</td>
          <td>24.854077</td>
          <td>0.089250</td>
          <td>24.200206</td>
          <td>0.112606</td>
          <td>0.012483</td>
          <td>0.010594</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.498908</td>
          <td>0.774503</td>
          <td>26.682754</td>
          <td>0.161992</td>
          <td>26.465688</td>
          <td>0.118614</td>
          <td>26.736102</td>
          <td>0.238977</td>
          <td>25.845853</td>
          <td>0.209679</td>
          <td>25.926150</td>
          <td>0.464904</td>
          <td>0.050570</td>
          <td>0.031026</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.229093</td>
          <td>0.304436</td>
          <td>26.123629</td>
          <td>0.099861</td>
          <td>26.018517</td>
          <td>0.080146</td>
          <td>25.879201</td>
          <td>0.115264</td>
          <td>25.674482</td>
          <td>0.181516</td>
          <td>25.350265</td>
          <td>0.296962</td>
          <td>0.029544</td>
          <td>0.015822</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.451779</td>
          <td>0.363125</td>
          <td>26.847182</td>
          <td>0.186259</td>
          <td>26.492492</td>
          <td>0.121410</td>
          <td>26.678768</td>
          <td>0.227900</td>
          <td>25.791996</td>
          <td>0.200425</td>
          <td>25.524296</td>
          <td>0.341180</td>
          <td>0.133115</td>
          <td>0.094508</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.733226</td>
          <td>0.500531</td>
          <td>26.941168</td>
          <td>0.232541</td>
          <td>26.176382</td>
          <td>0.109128</td>
          <td>25.235829</td>
          <td>0.078246</td>
          <td>24.683984</td>
          <td>0.091111</td>
          <td>24.294096</td>
          <td>0.145246</td>
          <td>0.062253</td>
          <td>0.034179</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.933992</td>
          <td>0.504076</td>
          <td>26.553528</td>
          <td>0.150074</td>
          <td>26.338934</td>
          <td>0.201465</td>
          <td>25.978666</td>
          <td>0.272371</td>
          <td>25.053794</td>
          <td>0.272529</td>
          <td>0.013378</td>
          <td>0.009072</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.290264</td>
          <td>0.750701</td>
          <td>29.099136</td>
          <td>1.110072</td>
          <td>27.553851</td>
          <td>0.353243</td>
          <td>25.703181</td>
          <td>0.120764</td>
          <td>25.244796</td>
          <td>0.151844</td>
          <td>24.255072</td>
          <td>0.143831</td>
          <td>0.121011</td>
          <td>0.070842</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.312683</td>
          <td>0.747962</td>
          <td>28.348288</td>
          <td>0.676650</td>
          <td>27.547410</td>
          <td>0.341515</td>
          <td>26.663356</td>
          <td>0.263507</td>
          <td>25.766015</td>
          <td>0.228634</td>
          <td>25.436968</td>
          <td>0.369813</td>
          <td>0.007488</td>
          <td>0.003951</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.506747</td>
          <td>0.468783</td>
          <td>26.213894</td>
          <td>0.144056</td>
          <td>25.935381</td>
          <td>0.103014</td>
          <td>25.593552</td>
          <td>0.125316</td>
          <td>25.611820</td>
          <td>0.234384</td>
          <td>26.295939</td>
          <td>0.787005</td>
          <td>0.258556</td>
          <td>0.216492</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.058958</td>
          <td>0.657030</td>
          <td>26.562484</td>
          <td>0.178787</td>
          <td>25.441956</td>
          <td>0.060855</td>
          <td>25.131730</td>
          <td>0.076176</td>
          <td>24.731827</td>
          <td>0.101148</td>
          <td>24.457683</td>
          <td>0.177783</td>
          <td>0.175304</td>
          <td>0.121080</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.063238</td>
          <td>0.296752</td>
          <td>26.663293</td>
          <td>0.183027</td>
          <td>26.197000</td>
          <td>0.110224</td>
          <td>25.188238</td>
          <td>0.074399</td>
          <td>24.685533</td>
          <td>0.090503</td>
          <td>24.230519</td>
          <td>0.136395</td>
          <td>0.012483</td>
          <td>0.010594</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.737272</td>
          <td>0.501209</td>
          <td>26.633446</td>
          <td>0.179318</td>
          <td>26.411162</td>
          <td>0.133477</td>
          <td>25.979869</td>
          <td>0.149334</td>
          <td>25.826994</td>
          <td>0.241783</td>
          <td>25.697995</td>
          <td>0.454084</td>
          <td>0.050570</td>
          <td>0.031026</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.656337</td>
          <td>0.212853</td>
          <td>26.201602</td>
          <td>0.123388</td>
          <td>26.150239</td>
          <td>0.105968</td>
          <td>26.113885</td>
          <td>0.166788</td>
          <td>25.853844</td>
          <td>0.246256</td>
          <td>25.477591</td>
          <td>0.382307</td>
          <td>0.029544</td>
          <td>0.015822</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.632801</td>
          <td>0.475222</td>
          <td>27.237364</td>
          <td>0.304814</td>
          <td>26.586566</td>
          <td>0.160948</td>
          <td>26.440742</td>
          <td>0.228727</td>
          <td>25.467359</td>
          <td>0.185498</td>
          <td>25.444265</td>
          <td>0.386743</td>
          <td>0.133115</td>
          <td>0.094508</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto

.. parsed-literal::

    




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.055751</td>
          <td>0.270332</td>
          <td>26.535482</td>
          <td>0.146841</td>
          <td>26.014565</td>
          <td>0.082544</td>
          <td>25.077176</td>
          <td>0.058865</td>
          <td>24.589321</td>
          <td>0.073028</td>
          <td>24.044553</td>
          <td>0.101676</td>
          <td>0.062253</td>
          <td>0.034179</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.588766</td>
          <td>0.342381</td>
          <td>26.662511</td>
          <td>0.140897</td>
          <td>26.046079</td>
          <td>0.133469</td>
          <td>26.030152</td>
          <td>0.244753</td>
          <td>25.208542</td>
          <td>0.265175</td>
          <td>0.013378</td>
          <td>0.009072</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.327274</td>
          <td>0.645577</td>
          <td>28.897259</td>
          <td>0.865849</td>
          <td>25.860062</td>
          <td>0.127664</td>
          <td>25.019231</td>
          <td>0.115660</td>
          <td>24.180299</td>
          <td>0.124521</td>
          <td>0.121011</td>
          <td>0.070842</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.362298</td>
          <td>0.707208</td>
          <td>30.442824</td>
          <td>1.968005</td>
          <td>27.626118</td>
          <td>0.314286</td>
          <td>26.390087</td>
          <td>0.178962</td>
          <td>25.957844</td>
          <td>0.230280</td>
          <td>24.669692</td>
          <td>0.168884</td>
          <td>0.007488</td>
          <td>0.003951</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.742565</td>
          <td>0.610004</td>
          <td>26.005511</td>
          <td>0.136111</td>
          <td>26.006580</td>
          <td>0.125102</td>
          <td>25.568712</td>
          <td>0.140247</td>
          <td>25.463122</td>
          <td>0.234956</td>
          <td>24.678612</td>
          <td>0.265369</td>
          <td>0.258556</td>
          <td>0.216492</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>28.691834</td>
          <td>1.690241</td>
          <td>26.132460</td>
          <td>0.123780</td>
          <td>25.408128</td>
          <td>0.059101</td>
          <td>25.063375</td>
          <td>0.071756</td>
          <td>24.790692</td>
          <td>0.106567</td>
          <td>24.547691</td>
          <td>0.191955</td>
          <td>0.175304</td>
          <td>0.121080</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.305362</td>
          <td>0.323912</td>
          <td>26.416200</td>
          <td>0.129024</td>
          <td>25.952472</td>
          <td>0.075744</td>
          <td>25.164999</td>
          <td>0.061578</td>
          <td>24.843175</td>
          <td>0.088559</td>
          <td>24.231037</td>
          <td>0.115889</td>
          <td>0.012483</td>
          <td>0.010594</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.061245</td>
          <td>0.580994</td>
          <td>26.648153</td>
          <td>0.160399</td>
          <td>26.337725</td>
          <td>0.108572</td>
          <td>26.110730</td>
          <td>0.144276</td>
          <td>26.992735</td>
          <td>0.528594</td>
          <td>25.456799</td>
          <td>0.330532</td>
          <td>0.050570</td>
          <td>0.031026</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.622829</td>
          <td>0.416358</td>
          <td>26.149540</td>
          <td>0.102820</td>
          <td>26.123722</td>
          <td>0.088601</td>
          <td>25.824221</td>
          <td>0.110739</td>
          <td>25.361178</td>
          <td>0.139899</td>
          <td>24.917240</td>
          <td>0.209605</td>
          <td>0.029544</td>
          <td>0.015822</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.302887</td>
          <td>0.739041</td>
          <td>26.836172</td>
          <td>0.209814</td>
          <td>26.733845</td>
          <td>0.173305</td>
          <td>26.200948</td>
          <td>0.177475</td>
          <td>26.027149</td>
          <td>0.280620</td>
          <td>31.693264</td>
          <td>5.273091</td>
          <td>0.133115</td>
          <td>0.094508</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
