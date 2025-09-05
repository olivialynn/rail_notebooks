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

    <pzflow.flow.Flow at 0x7ffa7b06bfd0>



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
    0      23.994413  0.071609  0.045861  
    1      25.391064  0.170323  0.157358  
    2      24.304707  0.052427  0.044419  
    3      25.291103  0.076666  0.045720  
    4      25.096743  0.023017  0.012375  
    ...          ...       ...       ...  
    99995  24.737946  0.155127  0.154149  
    99996  24.224169  0.127646  0.090037  
    99997  25.613836  0.089827  0.057359  
    99998  25.274899  0.011442  0.006364  
    99999  25.699642  0.088048  0.053488  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>26.450191</td>
          <td>0.132666</td>
          <td>25.919187</td>
          <td>0.073413</td>
          <td>25.202566</td>
          <td>0.063541</td>
          <td>24.581020</td>
          <td>0.070138</td>
          <td>23.975602</td>
          <td>0.092509</td>
          <td>0.071609</td>
          <td>0.045861</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.309050</td>
          <td>0.273234</td>
          <td>26.660268</td>
          <td>0.140381</td>
          <td>26.090646</td>
          <td>0.138452</td>
          <td>25.827422</td>
          <td>0.206470</td>
          <td>25.105289</td>
          <td>0.243220</td>
          <td>0.170323</td>
          <td>0.157358</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.990802</td>
          <td>1.052739</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.097403</td>
          <td>0.453031</td>
          <td>25.882374</td>
          <td>0.115583</td>
          <td>24.890475</td>
          <td>0.092152</td>
          <td>24.277184</td>
          <td>0.120409</td>
          <td>0.052427</td>
          <td>0.044419</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.716390</td>
          <td>1.556268</td>
          <td>28.226917</td>
          <td>0.554151</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.736556</td>
          <td>0.239067</td>
          <td>25.493699</td>
          <td>0.155617</td>
          <td>24.953909</td>
          <td>0.214518</td>
          <td>0.076666</td>
          <td>0.045720</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.910659</td>
          <td>0.234925</td>
          <td>26.425212</td>
          <td>0.129832</td>
          <td>26.027254</td>
          <td>0.080766</td>
          <td>25.629285</td>
          <td>0.092625</td>
          <td>25.382528</td>
          <td>0.141446</td>
          <td>25.408362</td>
          <td>0.311138</td>
          <td>0.023017</td>
          <td>0.012375</td>
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
          <td>26.775377</td>
          <td>0.465157</td>
          <td>26.525032</td>
          <td>0.141510</td>
          <td>25.493756</td>
          <td>0.050342</td>
          <td>25.182667</td>
          <td>0.062430</td>
          <td>24.807219</td>
          <td>0.085644</td>
          <td>24.778994</td>
          <td>0.185203</td>
          <td>0.155127</td>
          <td>0.154149</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.668631</td>
          <td>0.429186</td>
          <td>26.873131</td>
          <td>0.190382</td>
          <td>26.061363</td>
          <td>0.083233</td>
          <td>25.105870</td>
          <td>0.058318</td>
          <td>24.691461</td>
          <td>0.077332</td>
          <td>24.027199</td>
          <td>0.096795</td>
          <td>0.127646</td>
          <td>0.090037</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.034712</td>
          <td>1.080194</td>
          <td>26.586075</td>
          <td>0.149131</td>
          <td>26.412739</td>
          <td>0.113270</td>
          <td>26.111959</td>
          <td>0.141019</td>
          <td>25.739105</td>
          <td>0.191702</td>
          <td>25.389147</td>
          <td>0.306386</td>
          <td>0.089827</td>
          <td>0.057359</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.199319</td>
          <td>0.632127</td>
          <td>26.110304</td>
          <td>0.098703</td>
          <td>25.974641</td>
          <td>0.077101</td>
          <td>26.312745</td>
          <td>0.167495</td>
          <td>25.744008</td>
          <td>0.192496</td>
          <td>25.782391</td>
          <td>0.416983</td>
          <td>0.011442</td>
          <td>0.006364</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.081995</td>
          <td>1.110221</td>
          <td>27.255142</td>
          <td>0.261486</td>
          <td>26.586173</td>
          <td>0.131681</td>
          <td>26.475326</td>
          <td>0.192238</td>
          <td>26.132055</td>
          <td>0.265645</td>
          <td>25.611912</td>
          <td>0.365494</td>
          <td>0.088048</td>
          <td>0.053488</td>
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
          <td>28.119256</td>
          <td>1.229595</td>
          <td>26.591677</td>
          <td>0.174016</td>
          <td>25.992003</td>
          <td>0.093198</td>
          <td>25.164679</td>
          <td>0.073760</td>
          <td>24.654671</td>
          <td>0.089118</td>
          <td>23.953310</td>
          <td>0.108503</td>
          <td>0.071609</td>
          <td>0.045861</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.375987</td>
          <td>0.351859</td>
          <td>26.518580</td>
          <td>0.157984</td>
          <td>26.478386</td>
          <td>0.245428</td>
          <td>25.892035</td>
          <td>0.274182</td>
          <td>25.416241</td>
          <td>0.392558</td>
          <td>0.170323</td>
          <td>0.157358</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.107970</td>
          <td>0.575115</td>
          <td>29.572063</td>
          <td>1.330015</td>
          <td>25.945711</td>
          <td>0.145313</td>
          <td>24.982836</td>
          <td>0.118256</td>
          <td>24.426915</td>
          <td>0.162664</td>
          <td>0.052427</td>
          <td>0.044419</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.772497</td>
          <td>0.451387</td>
          <td>27.130472</td>
          <td>0.246933</td>
          <td>26.088960</td>
          <td>0.165168</td>
          <td>25.450064</td>
          <td>0.177623</td>
          <td>25.861192</td>
          <td>0.515892</td>
          <td>0.076666</td>
          <td>0.045720</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.820790</td>
          <td>0.243766</td>
          <td>26.268810</td>
          <td>0.130691</td>
          <td>26.025835</td>
          <td>0.094959</td>
          <td>25.916554</td>
          <td>0.140735</td>
          <td>25.349430</td>
          <td>0.161132</td>
          <td>25.448980</td>
          <td>0.373645</td>
          <td>0.023017</td>
          <td>0.012375</td>
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
          <td>27.392522</td>
          <td>0.823000</td>
          <td>26.327851</td>
          <td>0.146899</td>
          <td>25.503369</td>
          <td>0.064527</td>
          <td>25.040621</td>
          <td>0.070587</td>
          <td>24.724788</td>
          <td>0.100943</td>
          <td>24.613362</td>
          <td>0.203551</td>
          <td>0.155127</td>
          <td>0.154149</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.276198</td>
          <td>0.746998</td>
          <td>26.479785</td>
          <td>0.162092</td>
          <td>26.022980</td>
          <td>0.098441</td>
          <td>25.247153</td>
          <td>0.081637</td>
          <td>24.740435</td>
          <td>0.098770</td>
          <td>24.053866</td>
          <td>0.121799</td>
          <td>0.127646</td>
          <td>0.090037</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.379000</td>
          <td>0.385595</td>
          <td>26.882267</td>
          <td>0.223435</td>
          <td>26.562026</td>
          <td>0.153921</td>
          <td>26.039922</td>
          <td>0.159285</td>
          <td>25.986289</td>
          <td>0.278780</td>
          <td>25.606955</td>
          <td>0.428748</td>
          <td>0.089827</td>
          <td>0.057359</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.501246</td>
          <td>0.418341</td>
          <td>26.312092</td>
          <td>0.135562</td>
          <td>26.298239</td>
          <td>0.120361</td>
          <td>25.862820</td>
          <td>0.134241</td>
          <td>25.944963</td>
          <td>0.264953</td>
          <td>25.600024</td>
          <td>0.419477</td>
          <td>0.011442</td>
          <td>0.006364</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.365341</td>
          <td>0.381221</td>
          <td>26.688350</td>
          <td>0.189758</td>
          <td>26.691681</td>
          <td>0.171727</td>
          <td>26.172642</td>
          <td>0.178119</td>
          <td>25.606511</td>
          <td>0.203519</td>
          <td>25.922445</td>
          <td>0.541442</td>
          <td>0.088048</td>
          <td>0.053488</td>
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
          <td>29.339824</td>
          <td>2.091802</td>
          <td>26.580073</td>
          <td>0.154352</td>
          <td>26.257608</td>
          <td>0.103593</td>
          <td>25.219394</td>
          <td>0.067744</td>
          <td>24.556202</td>
          <td>0.071900</td>
          <td>23.928136</td>
          <td>0.093112</td>
          <td>0.071609</td>
          <td>0.045861</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.038804</td>
          <td>1.123012</td>
          <td>26.767287</td>
          <td>0.199983</td>
          <td>26.380673</td>
          <td>0.232126</td>
          <td>25.701577</td>
          <td>0.240286</td>
          <td>26.361861</td>
          <td>0.789608</td>
          <td>0.170323</td>
          <td>0.157358</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.579951</td>
          <td>0.311430</td>
          <td>26.028839</td>
          <td>0.135526</td>
          <td>25.234851</td>
          <td>0.128361</td>
          <td>24.577986</td>
          <td>0.161063</td>
          <td>0.052427</td>
          <td>0.044419</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.649115</td>
          <td>0.372941</td>
          <td>27.215152</td>
          <td>0.235681</td>
          <td>26.258776</td>
          <td>0.168426</td>
          <td>25.572759</td>
          <td>0.174864</td>
          <td>25.055942</td>
          <td>0.245339</td>
          <td>0.076666</td>
          <td>0.045720</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.676004</td>
          <td>0.193793</td>
          <td>25.995168</td>
          <td>0.089581</td>
          <td>25.978358</td>
          <td>0.077714</td>
          <td>25.634505</td>
          <td>0.093503</td>
          <td>25.647863</td>
          <td>0.178258</td>
          <td>24.855139</td>
          <td>0.198390</td>
          <td>0.023017</td>
          <td>0.012375</td>
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
          <td>29.332279</td>
          <td>2.234111</td>
          <td>26.159103</td>
          <td>0.128303</td>
          <td>25.386223</td>
          <td>0.058827</td>
          <td>24.947197</td>
          <td>0.065747</td>
          <td>24.980247</td>
          <td>0.127500</td>
          <td>25.226361</td>
          <td>0.339237</td>
          <td>0.155127</td>
          <td>0.154149</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.743839</td>
          <td>1.666975</td>
          <td>27.011563</td>
          <td>0.240438</td>
          <td>26.080773</td>
          <td>0.097469</td>
          <td>25.180253</td>
          <td>0.072238</td>
          <td>24.865876</td>
          <td>0.103767</td>
          <td>24.198412</td>
          <td>0.129813</td>
          <td>0.127646</td>
          <td>0.090037</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.137736</td>
          <td>1.183779</td>
          <td>26.885865</td>
          <td>0.204189</td>
          <td>26.487979</td>
          <td>0.129686</td>
          <td>26.398605</td>
          <td>0.193458</td>
          <td>26.610816</td>
          <td>0.413666</td>
          <td>25.695356</td>
          <td>0.415822</td>
          <td>0.089827</td>
          <td>0.057359</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.860802</td>
          <td>0.225603</td>
          <td>26.138475</td>
          <td>0.101268</td>
          <td>26.043992</td>
          <td>0.082063</td>
          <td>25.813437</td>
          <td>0.108972</td>
          <td>25.598933</td>
          <td>0.170433</td>
          <td>25.723615</td>
          <td>0.399015</td>
          <td>0.011442</td>
          <td>0.006364</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.103032</td>
          <td>1.158492</td>
          <td>26.398471</td>
          <td>0.134332</td>
          <td>26.562876</td>
          <td>0.137755</td>
          <td>26.154646</td>
          <td>0.156552</td>
          <td>25.891587</td>
          <td>0.231919</td>
          <td>25.236460</td>
          <td>0.288551</td>
          <td>0.088048</td>
          <td>0.053488</td>
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
