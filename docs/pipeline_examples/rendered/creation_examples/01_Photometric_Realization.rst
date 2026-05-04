Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``00_Quick_Start_in_Creation.ipynb``

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`Photometric_Realization.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization.ipynb>`__
in the ``interactive_examples/creation_examples/`` folder for a version
of this notebook in interactive mode.

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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f50921d5b70>



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
    0      23.994413  0.025381  0.015054  
    1      25.391064  0.031799  0.019731  
    2      24.304707  0.040078  0.030492  
    3      25.291103  0.092628  0.080336  
    4      25.096743  0.080754  0.069482  
    ...          ...       ...       ...  
    99995  24.737946  0.039499  0.036498  
    99996  24.224169  0.234147  0.143150  
    99997  25.613836  0.027061  0.014410  
    99998  25.274899  0.010843  0.010391  
    99999  25.699642  0.121999  0.094704  
    
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

    Inserting handle into data store.  output_truth: None, error_model
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
          <td>26.671598</td>
          <td>0.160457</td>
          <td>26.095985</td>
          <td>0.085811</td>
          <td>25.227412</td>
          <td>0.064956</td>
          <td>24.664496</td>
          <td>0.075511</td>
          <td>23.989313</td>
          <td>0.093630</td>
          <td>0.025381</td>
          <td>0.015054</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>39.774229</td>
          <td>12.318070</td>
          <td>27.133775</td>
          <td>0.236662</td>
          <td>26.555438</td>
          <td>0.128224</td>
          <td>26.201185</td>
          <td>0.152261</td>
          <td>25.934961</td>
          <td>0.225845</td>
          <td>25.365051</td>
          <td>0.300515</td>
          <td>0.031799</td>
          <td>0.019731</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.499104</td>
          <td>1.245311</td>
          <td>27.679857</td>
          <td>0.327890</td>
          <td>25.798476</td>
          <td>0.107427</td>
          <td>25.090827</td>
          <td>0.109828</td>
          <td>24.374351</td>
          <td>0.130994</td>
          <td>0.040078</td>
          <td>0.030492</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.821614</td>
          <td>0.950895</td>
          <td>27.955181</td>
          <td>0.453587</td>
          <td>27.347880</td>
          <td>0.250710</td>
          <td>26.128522</td>
          <td>0.143045</td>
          <td>25.432670</td>
          <td>0.147681</td>
          <td>25.573889</td>
          <td>0.354768</td>
          <td>0.092628</td>
          <td>0.080336</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.369638</td>
          <td>0.340441</td>
          <td>26.096228</td>
          <td>0.097494</td>
          <td>26.014476</td>
          <td>0.079861</td>
          <td>25.553972</td>
          <td>0.086688</td>
          <td>25.224638</td>
          <td>0.123394</td>
          <td>25.106477</td>
          <td>0.243458</td>
          <td>0.080754</td>
          <td>0.069482</td>
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
          <td>26.858762</td>
          <td>0.494909</td>
          <td>26.502750</td>
          <td>0.138820</td>
          <td>25.378052</td>
          <td>0.045428</td>
          <td>25.049807</td>
          <td>0.055487</td>
          <td>24.909717</td>
          <td>0.093723</td>
          <td>24.724539</td>
          <td>0.176856</td>
          <td>0.039499</td>
          <td>0.036498</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.913838</td>
          <td>0.515372</td>
          <td>26.806432</td>
          <td>0.179950</td>
          <td>25.883945</td>
          <td>0.071159</td>
          <td>25.152739</td>
          <td>0.060794</td>
          <td>24.778831</td>
          <td>0.083529</td>
          <td>24.089960</td>
          <td>0.102267</td>
          <td>0.234147</td>
          <td>0.143150</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.989461</td>
          <td>0.544542</td>
          <td>26.706713</td>
          <td>0.165335</td>
          <td>26.451774</td>
          <td>0.117187</td>
          <td>26.097705</td>
          <td>0.139297</td>
          <td>25.883728</td>
          <td>0.216418</td>
          <td>25.369028</td>
          <td>0.301478</td>
          <td>0.027061</td>
          <td>0.014410</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.921996</td>
          <td>0.237133</td>
          <td>26.182028</td>
          <td>0.105092</td>
          <td>26.145627</td>
          <td>0.089643</td>
          <td>25.907008</td>
          <td>0.118088</td>
          <td>25.543643</td>
          <td>0.162406</td>
          <td>24.841781</td>
          <td>0.195276</td>
          <td>0.010843</td>
          <td>0.010391</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.076161</td>
          <td>0.579532</td>
          <td>27.088971</td>
          <td>0.228046</td>
          <td>26.410967</td>
          <td>0.113095</td>
          <td>26.573472</td>
          <td>0.208754</td>
          <td>26.270909</td>
          <td>0.297296</td>
          <td>25.841269</td>
          <td>0.436099</td>
          <td>0.121999</td>
          <td>0.094704</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model_gaap


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
          <td>inf</td>
          <td>inf</td>
          <td>26.956624</td>
          <td>0.234133</td>
          <td>26.132400</td>
          <td>0.104283</td>
          <td>25.073269</td>
          <td>0.067277</td>
          <td>24.775971</td>
          <td>0.098080</td>
          <td>24.059550</td>
          <td>0.117739</td>
          <td>0.025381</td>
          <td>0.015054</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.948611</td>
          <td>0.232764</td>
          <td>26.422614</td>
          <td>0.134331</td>
          <td>26.475199</td>
          <td>0.226168</td>
          <td>26.098632</td>
          <td>0.300658</td>
          <td>26.418167</td>
          <td>0.754675</td>
          <td>0.031799</td>
          <td>0.019731</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.733621</td>
          <td>0.808914</td>
          <td>26.153529</td>
          <td>0.172924</td>
          <td>25.057966</td>
          <td>0.125768</td>
          <td>24.243685</td>
          <td>0.138489</td>
          <td>0.040078</td>
          <td>0.030492</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.886448</td>
          <td>0.974392</td>
          <td>27.370655</td>
          <td>0.303424</td>
          <td>26.440918</td>
          <td>0.224724</td>
          <td>25.811299</td>
          <td>0.242962</td>
          <td>25.109226</td>
          <td>0.291741</td>
          <td>0.092628</td>
          <td>0.080336</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.688576</td>
          <td>0.487550</td>
          <td>26.203830</td>
          <td>0.125520</td>
          <td>26.035343</td>
          <td>0.097472</td>
          <td>25.771703</td>
          <td>0.126447</td>
          <td>25.403309</td>
          <td>0.171637</td>
          <td>26.103114</td>
          <td>0.616620</td>
          <td>0.080754</td>
          <td>0.069482</td>
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
          <td>26.295942</td>
          <td>0.358038</td>
          <td>26.472929</td>
          <td>0.156279</td>
          <td>25.457224</td>
          <td>0.057686</td>
          <td>25.129194</td>
          <td>0.070944</td>
          <td>25.198415</td>
          <td>0.142078</td>
          <td>24.847796</td>
          <td>0.231102</td>
          <td>0.039499</td>
          <td>0.036498</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.557070</td>
          <td>0.470223</td>
          <td>27.054711</td>
          <td>0.278616</td>
          <td>25.878795</td>
          <td>0.093199</td>
          <td>25.384693</td>
          <td>0.099203</td>
          <td>25.004229</td>
          <td>0.133419</td>
          <td>24.080885</td>
          <td>0.134047</td>
          <td>0.234147</td>
          <td>0.143150</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.138832</td>
          <td>0.665531</td>
          <td>26.984267</td>
          <td>0.239564</td>
          <td>26.345269</td>
          <td>0.125537</td>
          <td>26.472421</td>
          <td>0.225475</td>
          <td>26.194667</td>
          <td>0.324426</td>
          <td>25.444051</td>
          <td>0.372361</td>
          <td>0.027061</td>
          <td>0.014410</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.092498</td>
          <td>0.644093</td>
          <td>26.347806</td>
          <td>0.139811</td>
          <td>26.156742</td>
          <td>0.106410</td>
          <td>26.084063</td>
          <td>0.162350</td>
          <td>25.466637</td>
          <td>0.177898</td>
          <td>25.233967</td>
          <td>0.315123</td>
          <td>0.010843</td>
          <td>0.010391</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.466170</td>
          <td>0.845194</td>
          <td>26.452783</td>
          <td>0.158289</td>
          <td>26.255951</td>
          <td>0.120541</td>
          <td>26.509751</td>
          <td>0.241160</td>
          <td>26.110156</td>
          <td>0.313722</td>
          <td>25.805307</td>
          <td>0.506125</td>
          <td>0.121999</td>
          <td>0.094704</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model_auto


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
          <td>29.009784</td>
          <td>1.790241</td>
          <td>26.552899</td>
          <td>0.145667</td>
          <td>26.049812</td>
          <td>0.082877</td>
          <td>25.186154</td>
          <td>0.063015</td>
          <td>24.659750</td>
          <td>0.075641</td>
          <td>24.002871</td>
          <td>0.095331</td>
          <td>0.025381</td>
          <td>0.015054</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.899450</td>
          <td>1.001431</td>
          <td>27.487276</td>
          <td>0.317788</td>
          <td>26.453314</td>
          <td>0.118443</td>
          <td>26.521166</td>
          <td>0.201692</td>
          <td>26.006025</td>
          <td>0.241657</td>
          <td>25.312828</td>
          <td>0.290728</td>
          <td>0.031799</td>
          <td>0.019731</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.550544</td>
          <td>0.808129</td>
          <td>27.893695</td>
          <td>0.438515</td>
          <td>29.447117</td>
          <td>1.128097</td>
          <td>26.080813</td>
          <td>0.139696</td>
          <td>25.030807</td>
          <td>0.105986</td>
          <td>24.400744</td>
          <td>0.136353</td>
          <td>0.040078</td>
          <td>0.030492</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.092817</td>
          <td>0.618311</td>
          <td>29.279184</td>
          <td>1.159839</td>
          <td>27.478922</td>
          <td>0.303748</td>
          <td>26.449654</td>
          <td>0.206353</td>
          <td>25.410810</td>
          <td>0.158607</td>
          <td>25.532178</td>
          <td>0.373913</td>
          <td>0.092628</td>
          <td>0.080336</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.291869</td>
          <td>0.335046</td>
          <td>26.283117</td>
          <td>0.122101</td>
          <td>25.885570</td>
          <td>0.076580</td>
          <td>25.656552</td>
          <td>0.102221</td>
          <td>25.186716</td>
          <td>0.128115</td>
          <td>25.155226</td>
          <td>0.271475</td>
          <td>0.080754</td>
          <td>0.069482</td>
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
          <td>27.264822</td>
          <td>0.668480</td>
          <td>26.260371</td>
          <td>0.114409</td>
          <td>25.503827</td>
          <td>0.051796</td>
          <td>25.134553</td>
          <td>0.061057</td>
          <td>24.796162</td>
          <td>0.086468</td>
          <td>24.652979</td>
          <td>0.169674</td>
          <td>0.039499</td>
          <td>0.036498</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.579019</td>
          <td>0.198526</td>
          <td>26.091741</td>
          <td>0.119314</td>
          <td>25.118702</td>
          <td>0.083614</td>
          <td>24.995904</td>
          <td>0.140679</td>
          <td>24.378566</td>
          <td>0.183715</td>
          <td>0.234147</td>
          <td>0.143150</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.578919</td>
          <td>0.402285</td>
          <td>26.733363</td>
          <td>0.170032</td>
          <td>26.283661</td>
          <td>0.101831</td>
          <td>26.092725</td>
          <td>0.139609</td>
          <td>26.176174</td>
          <td>0.276979</td>
          <td>24.994443</td>
          <td>0.223274</td>
          <td>0.027061</td>
          <td>0.014410</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.419370</td>
          <td>0.735285</td>
          <td>26.250616</td>
          <td>0.111719</td>
          <td>26.197404</td>
          <td>0.093961</td>
          <td>25.979990</td>
          <td>0.126016</td>
          <td>26.386267</td>
          <td>0.326492</td>
          <td>25.356947</td>
          <td>0.298998</td>
          <td>0.010843</td>
          <td>0.010391</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.862942</td>
          <td>0.539642</td>
          <td>27.392824</td>
          <td>0.326905</td>
          <td>26.572456</td>
          <td>0.149097</td>
          <td>26.659863</td>
          <td>0.256985</td>
          <td>25.816202</td>
          <td>0.233237</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.121999</td>
          <td>0.094704</td>
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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_22_0.png


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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_23_0.png


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
