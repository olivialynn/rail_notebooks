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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.19/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f2c6b9632e0>



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
    0      23.994413  0.148667  0.132399  
    1      25.391064  0.117294  0.077764  
    2      24.304707  0.020791  0.015261  
    3      25.291103  0.066139  0.044894  
    4      25.096743  0.148511  0.086048  
    ...          ...       ...       ...  
    99995  24.737946  0.002191  0.001500  
    99996  24.224169  0.057329  0.039035  
    99997  25.613836  0.098662  0.069579  
    99998  25.274899  0.001780  0.001095  
    99999  25.699642  0.151464  0.120282  
    
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
          <td>26.807044</td>
          <td>0.476283</td>
          <td>26.883573</td>
          <td>0.192065</td>
          <td>25.872487</td>
          <td>0.070441</td>
          <td>25.109794</td>
          <td>0.058521</td>
          <td>24.659206</td>
          <td>0.075159</td>
          <td>24.010211</td>
          <td>0.095363</td>
          <td>0.148667</td>
          <td>0.132399</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.447399</td>
          <td>0.748556</td>
          <td>28.471194</td>
          <td>0.658429</td>
          <td>26.549011</td>
          <td>0.127512</td>
          <td>26.102483</td>
          <td>0.139872</td>
          <td>25.731768</td>
          <td>0.190520</td>
          <td>25.313337</td>
          <td>0.288246</td>
          <td>0.117294</td>
          <td>0.077764</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.576915</td>
          <td>2.264814</td>
          <td>29.259783</td>
          <td>1.087884</td>
          <td>27.390172</td>
          <td>0.259557</td>
          <td>25.989539</td>
          <td>0.126862</td>
          <td>25.047729</td>
          <td>0.105771</td>
          <td>24.403947</td>
          <td>0.134389</td>
          <td>0.020791</td>
          <td>0.015261</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.853307</td>
          <td>0.419886</td>
          <td>26.965450</td>
          <td>0.182204</td>
          <td>26.221994</td>
          <td>0.155001</td>
          <td>25.706824</td>
          <td>0.186550</td>
          <td>25.258402</td>
          <td>0.275695</td>
          <td>0.066139</td>
          <td>0.044894</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.829491</td>
          <td>0.955496</td>
          <td>26.211194</td>
          <td>0.107802</td>
          <td>25.978120</td>
          <td>0.077338</td>
          <td>25.415724</td>
          <td>0.076736</td>
          <td>25.545481</td>
          <td>0.162660</td>
          <td>25.197346</td>
          <td>0.262311</td>
          <td>0.148511</td>
          <td>0.086048</td>
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
          <td>27.825846</td>
          <td>0.953365</td>
          <td>26.435233</td>
          <td>0.130962</td>
          <td>25.458844</td>
          <td>0.048806</td>
          <td>25.071091</td>
          <td>0.056545</td>
          <td>24.900401</td>
          <td>0.092960</td>
          <td>25.171915</td>
          <td>0.256908</td>
          <td>0.002191</td>
          <td>0.001500</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.601556</td>
          <td>1.469963</td>
          <td>26.800663</td>
          <td>0.179073</td>
          <td>26.258728</td>
          <td>0.099003</td>
          <td>25.224421</td>
          <td>0.064784</td>
          <td>24.833758</td>
          <td>0.087669</td>
          <td>24.258157</td>
          <td>0.118434</td>
          <td>0.057329</td>
          <td>0.039035</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.917364</td>
          <td>0.516705</td>
          <td>26.602445</td>
          <td>0.151239</td>
          <td>26.459639</td>
          <td>0.117992</td>
          <td>26.093549</td>
          <td>0.138799</td>
          <td>25.780270</td>
          <td>0.198461</td>
          <td>26.311603</td>
          <td>0.615125</td>
          <td>0.098662</td>
          <td>0.069579</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.797749</td>
          <td>0.472995</td>
          <td>26.345149</td>
          <td>0.121131</td>
          <td>26.086107</td>
          <td>0.085068</td>
          <td>26.003024</td>
          <td>0.128353</td>
          <td>25.781998</td>
          <td>0.198749</td>
          <td>25.753221</td>
          <td>0.407770</td>
          <td>0.001780</td>
          <td>0.001095</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.196745</td>
          <td>0.296632</td>
          <td>27.015577</td>
          <td>0.214540</td>
          <td>26.506237</td>
          <td>0.122868</td>
          <td>26.259184</td>
          <td>0.160012</td>
          <td>25.682097</td>
          <td>0.182690</td>
          <td>25.461152</td>
          <td>0.324523</td>
          <td>0.151464</td>
          <td>0.120282</td>
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
          <td>28.559501</td>
          <td>1.580562</td>
          <td>26.530391</td>
          <td>0.172759</td>
          <td>25.965046</td>
          <td>0.095782</td>
          <td>25.196833</td>
          <td>0.080016</td>
          <td>24.780251</td>
          <td>0.104687</td>
          <td>23.986143</td>
          <td>0.117602</td>
          <td>0.148667</td>
          <td>0.132399</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.056548</td>
          <td>0.640527</td>
          <td>27.361808</td>
          <td>0.333633</td>
          <td>26.512463</td>
          <td>0.149481</td>
          <td>26.393742</td>
          <td>0.217666</td>
          <td>26.121153</td>
          <td>0.314637</td>
          <td>25.261500</td>
          <td>0.331854</td>
          <td>0.117294</td>
          <td>0.077764</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.532237</td>
          <td>0.863209</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.954646</td>
          <td>0.467609</td>
          <td>25.947762</td>
          <td>0.144563</td>
          <td>25.156570</td>
          <td>0.136538</td>
          <td>24.334127</td>
          <td>0.149218</td>
          <td>0.020791</td>
          <td>0.015261</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.424478</td>
          <td>0.809844</td>
          <td>29.352793</td>
          <td>1.263912</td>
          <td>28.103949</td>
          <td>0.526385</td>
          <td>26.593715</td>
          <td>0.251463</td>
          <td>25.628800</td>
          <td>0.205990</td>
          <td>25.421989</td>
          <td>0.369077</td>
          <td>0.066139</td>
          <td>0.044894</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.784834</td>
          <td>0.244664</td>
          <td>26.212054</td>
          <td>0.129659</td>
          <td>25.999403</td>
          <td>0.097153</td>
          <td>25.769571</td>
          <td>0.129898</td>
          <td>25.437710</td>
          <td>0.181604</td>
          <td>25.230755</td>
          <td>0.328306</td>
          <td>0.148511</td>
          <td>0.086048</td>
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
          <td>28.236590</td>
          <td>1.303109</td>
          <td>26.253867</td>
          <td>0.128882</td>
          <td>25.471050</td>
          <td>0.058109</td>
          <td>25.022268</td>
          <td>0.064207</td>
          <td>24.700059</td>
          <td>0.091625</td>
          <td>25.045738</td>
          <td>0.270636</td>
          <td>0.002191</td>
          <td>0.001500</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.682622</td>
          <td>0.951726</td>
          <td>26.625499</td>
          <td>0.178451</td>
          <td>26.053205</td>
          <td>0.097945</td>
          <td>25.360111</td>
          <td>0.087270</td>
          <td>24.727520</td>
          <td>0.094625</td>
          <td>24.388715</td>
          <td>0.157463</td>
          <td>0.057329</td>
          <td>0.039035</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.578301</td>
          <td>0.899922</td>
          <td>26.617877</td>
          <td>0.179801</td>
          <td>26.406110</td>
          <td>0.135296</td>
          <td>26.080261</td>
          <td>0.165735</td>
          <td>25.561005</td>
          <td>0.197102</td>
          <td>25.240698</td>
          <td>0.323912</td>
          <td>0.098662</td>
          <td>0.069579</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.097036</td>
          <td>0.304816</td>
          <td>26.121874</td>
          <td>0.114940</td>
          <td>26.325197</td>
          <td>0.123177</td>
          <td>25.820205</td>
          <td>0.129347</td>
          <td>25.365432</td>
          <td>0.163165</td>
          <td>24.567854</td>
          <td>0.181916</td>
          <td>0.001780</td>
          <td>0.001095</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.227875</td>
          <td>0.731869</td>
          <td>27.057079</td>
          <td>0.267126</td>
          <td>26.731758</td>
          <td>0.185009</td>
          <td>26.757768</td>
          <td>0.301064</td>
          <td>26.130390</td>
          <td>0.324878</td>
          <td>25.498764</td>
          <td>0.409348</td>
          <td>0.151464</td>
          <td>0.120282</td>
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
          <td>26.981254</td>
          <td>0.614959</td>
          <td>26.384546</td>
          <td>0.151015</td>
          <td>25.988458</td>
          <td>0.096657</td>
          <td>25.075097</td>
          <td>0.071009</td>
          <td>24.659229</td>
          <td>0.093088</td>
          <td>24.135629</td>
          <td>0.132333</td>
          <td>0.148667</td>
          <td>0.132399</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.702294</td>
          <td>0.937161</td>
          <td>27.609376</td>
          <td>0.380845</td>
          <td>26.401141</td>
          <td>0.125884</td>
          <td>26.381498</td>
          <td>0.199651</td>
          <td>25.396911</td>
          <td>0.160494</td>
          <td>24.929374</td>
          <td>0.235633</td>
          <td>0.117294</td>
          <td>0.077764</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.409227</td>
          <td>0.264737</td>
          <td>25.977352</td>
          <td>0.126115</td>
          <td>24.895311</td>
          <td>0.092962</td>
          <td>24.337110</td>
          <td>0.127423</td>
          <td>0.020791</td>
          <td>0.015261</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.068480</td>
          <td>0.590076</td>
          <td>28.813903</td>
          <td>0.849739</td>
          <td>27.681545</td>
          <td>0.340791</td>
          <td>26.160002</td>
          <td>0.153326</td>
          <td>25.856193</td>
          <td>0.219955</td>
          <td>25.316685</td>
          <td>0.300621</td>
          <td>0.066139</td>
          <td>0.044894</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.046110</td>
          <td>0.624122</td>
          <td>26.099786</td>
          <td>0.112855</td>
          <td>26.051400</td>
          <td>0.097049</td>
          <td>25.823591</td>
          <td>0.129768</td>
          <td>25.473737</td>
          <td>0.178994</td>
          <td>25.117973</td>
          <td>0.287069</td>
          <td>0.148511</td>
          <td>0.086048</td>
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
          <td>27.265328</td>
          <td>0.661747</td>
          <td>26.539366</td>
          <td>0.143271</td>
          <td>25.502840</td>
          <td>0.050752</td>
          <td>24.989645</td>
          <td>0.052603</td>
          <td>24.927221</td>
          <td>0.095179</td>
          <td>24.828619</td>
          <td>0.193133</td>
          <td>0.002191</td>
          <td>0.001500</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.090978</td>
          <td>0.596192</td>
          <td>26.962321</td>
          <td>0.210630</td>
          <td>26.021946</td>
          <td>0.082974</td>
          <td>25.161405</td>
          <td>0.063344</td>
          <td>24.865319</td>
          <td>0.093022</td>
          <td>24.447540</td>
          <td>0.144087</td>
          <td>0.057329</td>
          <td>0.039035</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.521129</td>
          <td>0.405273</td>
          <td>26.818013</td>
          <td>0.195923</td>
          <td>26.393871</td>
          <td>0.121726</td>
          <td>26.239538</td>
          <td>0.172260</td>
          <td>26.115671</td>
          <td>0.284653</td>
          <td>25.909005</td>
          <td>0.496342</td>
          <td>0.098662</td>
          <td>0.069579</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.154274</td>
          <td>0.612503</td>
          <td>26.016477</td>
          <td>0.090911</td>
          <td>26.014515</td>
          <td>0.079866</td>
          <td>25.989143</td>
          <td>0.126822</td>
          <td>26.100584</td>
          <td>0.258905</td>
          <td>25.039952</td>
          <td>0.230440</td>
          <td>0.001780</td>
          <td>0.001095</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.602104</td>
          <td>0.463830</td>
          <td>26.923544</td>
          <td>0.235616</td>
          <td>26.406234</td>
          <td>0.137551</td>
          <td>26.489259</td>
          <td>0.237550</td>
          <td>25.532268</td>
          <td>0.195495</td>
          <td>26.043602</td>
          <td>0.602028</td>
          <td>0.151464</td>
          <td>0.120282</td>
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
