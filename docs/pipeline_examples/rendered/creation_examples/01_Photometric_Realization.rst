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

    <pzflow.flow.Flow at 0x7f760e11d960>



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
    0      23.994413  0.050027  0.041408  
    1      25.391064  0.116972  0.114641  
    2      24.304707  0.029470  0.024759  
    3      25.291103  0.022677  0.011476  
    4      25.096743  0.176950  0.157100  
    ...          ...       ...       ...  
    99995  24.737946  0.048283  0.034678  
    99996  24.224169  0.035070  0.027178  
    99997  25.613836  0.039350  0.023025  
    99998  25.274899  0.136172  0.127666  
    99999  25.699642  0.017884  0.017135  
    
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
          <td>26.860206</td>
          <td>0.495437</td>
          <td>26.667893</td>
          <td>0.159950</td>
          <td>25.975682</td>
          <td>0.077171</td>
          <td>25.210115</td>
          <td>0.063968</td>
          <td>24.672547</td>
          <td>0.076050</td>
          <td>23.917291</td>
          <td>0.087885</td>
          <td>0.050027</td>
          <td>0.041408</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.213944</td>
          <td>1.196506</td>
          <td>27.395814</td>
          <td>0.293121</td>
          <td>26.821197</td>
          <td>0.161169</td>
          <td>26.055949</td>
          <td>0.134366</td>
          <td>25.630423</td>
          <td>0.174860</td>
          <td>25.885752</td>
          <td>0.451009</td>
          <td>0.116972</td>
          <td>0.114641</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.730231</td>
          <td>0.341238</td>
          <td>26.114781</td>
          <td>0.141363</td>
          <td>25.132171</td>
          <td>0.113860</td>
          <td>24.169777</td>
          <td>0.109657</td>
          <td>0.029470</td>
          <td>0.024759</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.337448</td>
          <td>0.331886</td>
          <td>27.594213</td>
          <td>0.343382</td>
          <td>28.136546</td>
          <td>0.466538</td>
          <td>26.305944</td>
          <td>0.166527</td>
          <td>25.650681</td>
          <td>0.177892</td>
          <td>24.944806</td>
          <td>0.212894</td>
          <td>0.022677</td>
          <td>0.011476</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.356837</td>
          <td>0.704416</td>
          <td>26.410209</td>
          <td>0.128158</td>
          <td>25.984829</td>
          <td>0.077797</td>
          <td>25.762133</td>
          <td>0.104067</td>
          <td>25.170634</td>
          <td>0.117737</td>
          <td>25.003480</td>
          <td>0.223562</td>
          <td>0.176950</td>
          <td>0.157100</td>
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
          <td>27.112849</td>
          <td>0.594842</td>
          <td>26.339644</td>
          <td>0.120554</td>
          <td>25.412704</td>
          <td>0.046847</td>
          <td>25.174688</td>
          <td>0.061989</td>
          <td>24.807856</td>
          <td>0.085692</td>
          <td>24.820500</td>
          <td>0.191806</td>
          <td>0.048283</td>
          <td>0.034678</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.856337</td>
          <td>0.187705</td>
          <td>26.047184</td>
          <td>0.082199</td>
          <td>25.149877</td>
          <td>0.060640</td>
          <td>24.751718</td>
          <td>0.081556</td>
          <td>24.319688</td>
          <td>0.124936</td>
          <td>0.035070</td>
          <td>0.027178</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.409304</td>
          <td>0.729758</td>
          <td>26.931148</td>
          <td>0.199905</td>
          <td>26.458081</td>
          <td>0.117832</td>
          <td>26.012296</td>
          <td>0.129387</td>
          <td>26.204104</td>
          <td>0.281677</td>
          <td>25.154421</td>
          <td>0.253250</td>
          <td>0.039350</td>
          <td>0.023025</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.356529</td>
          <td>0.336934</td>
          <td>26.194414</td>
          <td>0.106235</td>
          <td>26.201797</td>
          <td>0.094180</td>
          <td>25.794141</td>
          <td>0.107021</td>
          <td>25.490845</td>
          <td>0.155237</td>
          <td>25.151940</td>
          <td>0.252734</td>
          <td>0.136172</td>
          <td>0.127666</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.636632</td>
          <td>0.418856</td>
          <td>26.820839</td>
          <td>0.182158</td>
          <td>26.512945</td>
          <td>0.123585</td>
          <td>26.471318</td>
          <td>0.191589</td>
          <td>26.156702</td>
          <td>0.271036</td>
          <td>25.968737</td>
          <td>0.479921</td>
          <td>0.017884</td>
          <td>0.017135</td>
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
          <td>27.213190</td>
          <td>0.702486</td>
          <td>26.778138</td>
          <td>0.202774</td>
          <td>25.966258</td>
          <td>0.090658</td>
          <td>25.074921</td>
          <td>0.067775</td>
          <td>24.796676</td>
          <td>0.100440</td>
          <td>23.878768</td>
          <td>0.101139</td>
          <td>0.050027</td>
          <td>0.041408</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.855798</td>
          <td>0.491691</td>
          <td>26.814778</td>
          <td>0.195399</td>
          <td>26.309943</td>
          <td>0.205192</td>
          <td>25.902579</td>
          <td>0.266456</td>
          <td>25.424535</td>
          <td>0.381035</td>
          <td>0.116972</td>
          <td>0.114641</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.876592</td>
          <td>0.441468</td>
          <td>26.067575</td>
          <td>0.160425</td>
          <td>25.090271</td>
          <td>0.129110</td>
          <td>24.516992</td>
          <td>0.174674</td>
          <td>0.029470</td>
          <td>0.024759</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.879747</td>
          <td>0.554340</td>
          <td>29.548870</td>
          <td>1.395293</td>
          <td>28.054391</td>
          <td>0.503534</td>
          <td>26.383560</td>
          <td>0.209275</td>
          <td>25.349230</td>
          <td>0.161095</td>
          <td>25.414954</td>
          <td>0.363833</td>
          <td>0.022677</td>
          <td>0.011476</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.688290</td>
          <td>0.509592</td>
          <td>26.047332</td>
          <td>0.116504</td>
          <td>25.885702</td>
          <td>0.091490</td>
          <td>25.740958</td>
          <td>0.131938</td>
          <td>25.645045</td>
          <td>0.224447</td>
          <td>25.127751</td>
          <td>0.313800</td>
          <td>0.176950</td>
          <td>0.157100</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.306973</td>
          <td>0.135648</td>
          <td>25.434841</td>
          <td>0.056613</td>
          <td>25.193141</td>
          <td>0.075154</td>
          <td>24.783182</td>
          <td>0.099146</td>
          <td>24.921841</td>
          <td>0.245930</td>
          <td>0.048283</td>
          <td>0.034678</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.741398</td>
          <td>0.195974</td>
          <td>25.948743</td>
          <td>0.088935</td>
          <td>25.240840</td>
          <td>0.078172</td>
          <td>24.890272</td>
          <td>0.108594</td>
          <td>24.245702</td>
          <td>0.138596</td>
          <td>0.035070</td>
          <td>0.027178</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.713289</td>
          <td>1.659206</td>
          <td>26.678600</td>
          <td>0.185904</td>
          <td>26.371111</td>
          <td>0.128624</td>
          <td>26.504797</td>
          <td>0.232048</td>
          <td>25.953130</td>
          <td>0.267526</td>
          <td>25.632720</td>
          <td>0.431294</td>
          <td>0.039350</td>
          <td>0.023025</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>28.125926</td>
          <td>1.260370</td>
          <td>26.103268</td>
          <td>0.118925</td>
          <td>25.982047</td>
          <td>0.096527</td>
          <td>25.921575</td>
          <td>0.149402</td>
          <td>25.758174</td>
          <td>0.239409</td>
          <td>25.018399</td>
          <td>0.279125</td>
          <td>0.136172</td>
          <td>0.127666</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.887055</td>
          <td>1.071787</td>
          <td>27.141865</td>
          <td>0.272455</td>
          <td>26.598413</td>
          <td>0.156050</td>
          <td>25.916096</td>
          <td>0.140663</td>
          <td>26.391313</td>
          <td>0.378500</td>
          <td>25.972925</td>
          <td>0.553729</td>
          <td>0.017884</td>
          <td>0.017135</td>
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
          <td>26.745427</td>
          <td>0.462431</td>
          <td>26.895584</td>
          <td>0.198564</td>
          <td>25.966420</td>
          <td>0.078716</td>
          <td>25.145144</td>
          <td>0.062194</td>
          <td>24.723825</td>
          <td>0.081827</td>
          <td>24.115272</td>
          <td>0.107607</td>
          <td>0.050027</td>
          <td>0.041408</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.227778</td>
          <td>0.703592</td>
          <td>27.685168</td>
          <td>0.415781</td>
          <td>26.440051</td>
          <td>0.135107</td>
          <td>26.252782</td>
          <td>0.185971</td>
          <td>26.110819</td>
          <td>0.301006</td>
          <td>25.044201</td>
          <td>0.268478</td>
          <td>0.116972</td>
          <td>0.114641</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.635072</td>
          <td>0.357349</td>
          <td>27.871432</td>
          <td>0.384502</td>
          <td>26.039512</td>
          <td>0.133841</td>
          <td>25.106012</td>
          <td>0.112398</td>
          <td>24.249012</td>
          <td>0.118707</td>
          <td>0.029470</td>
          <td>0.024759</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.613907</td>
          <td>3.218714</td>
          <td>28.114706</td>
          <td>0.512316</td>
          <td>27.470066</td>
          <td>0.278147</td>
          <td>26.267092</td>
          <td>0.161816</td>
          <td>26.042433</td>
          <td>0.247840</td>
          <td>25.058932</td>
          <td>0.235086</td>
          <td>0.022677</td>
          <td>0.011476</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.370371</td>
          <td>0.409987</td>
          <td>26.614739</td>
          <td>0.194688</td>
          <td>26.197061</td>
          <td>0.123814</td>
          <td>25.843739</td>
          <td>0.148704</td>
          <td>25.659775</td>
          <td>0.233902</td>
          <td>25.675001</td>
          <td>0.491915</td>
          <td>0.176950</td>
          <td>0.157100</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.437628</td>
          <td>0.133886</td>
          <td>25.380505</td>
          <td>0.046618</td>
          <td>25.055161</td>
          <td>0.057153</td>
          <td>24.910735</td>
          <td>0.096024</td>
          <td>24.689233</td>
          <td>0.175708</td>
          <td>0.048283</td>
          <td>0.034678</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.950791</td>
          <td>0.533501</td>
          <td>26.652184</td>
          <td>0.159586</td>
          <td>26.046903</td>
          <td>0.083274</td>
          <td>25.277099</td>
          <td>0.068835</td>
          <td>25.029591</td>
          <td>0.105481</td>
          <td>24.493987</td>
          <td>0.147195</td>
          <td>0.035070</td>
          <td>0.027178</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.267196</td>
          <td>1.239802</td>
          <td>26.692958</td>
          <td>0.165332</td>
          <td>26.287318</td>
          <td>0.102928</td>
          <td>26.184148</td>
          <td>0.152201</td>
          <td>25.704843</td>
          <td>0.188735</td>
          <td>26.221800</td>
          <td>0.583991</td>
          <td>0.039350</td>
          <td>0.023025</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.816878</td>
          <td>0.248187</td>
          <td>26.189875</td>
          <td>0.125324</td>
          <td>26.150044</td>
          <td>0.109030</td>
          <td>26.141858</td>
          <td>0.175799</td>
          <td>25.879270</td>
          <td>0.258300</td>
          <td>25.045142</td>
          <td>0.278510</td>
          <td>0.136172</td>
          <td>0.127666</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.516345</td>
          <td>0.785106</td>
          <td>26.643387</td>
          <td>0.157185</td>
          <td>26.583759</td>
          <td>0.131945</td>
          <td>26.574119</td>
          <td>0.209736</td>
          <td>25.814007</td>
          <td>0.204971</td>
          <td>25.476101</td>
          <td>0.329685</td>
          <td>0.017884</td>
          <td>0.017135</td>
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
